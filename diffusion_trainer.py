import logging
import time
import gc

import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from scipy.stats import ttest_rel
from tqdm import tqdm
from ema import EMA
from model import *

from pretraining.dcg import DCG as AuxCls
from pretraining.resnet import ResNet18
from utils import *
from diffusion_utils import *
from tqdm import tqdm
plt.style.use('ggplot')

import wandb
import random

class Diffusion(object):
    def __init__(self, args, config, device=None):
        
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        self.num_timesteps = config.diffusion.timesteps
        self.test_num_timesteps = config.diffusion.test_timesteps
        self.vis_step = config.diffusion.vis_step
        self.num_figs = config.diffusion.num_figs

        betas = make_beta_schedule(schedule=config.diffusion.beta_schedule, num_timesteps=self.num_timesteps,
                                   start=config.diffusion.beta_start, end=config.diffusion.beta_end)
        betas = self.betas = betas.float().to(self.device)
        self.betas_sqrt = torch.sqrt(betas)
        alphas = 1.0 - betas
        self.alphas = alphas
        self.one_minus_betas_sqrt = torch.sqrt(alphas)
        alphas_cumprod = alphas.cumprod(dim=0)
        self.alphas_bar_sqrt = torch.sqrt(alphas_cumprod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_cumprod)
        if config.diffusion.beta_schedule == "cosine":
            self.one_minus_alphas_bar_sqrt *= 0.9999  # avoid division by 0 for 1/sqrt(alpha_bar_t) during inference
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.posterior_mean_coeff_1 = (
                betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_mean_coeff_2 = (
                torch.sqrt(alphas) * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        )
        posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_variance = posterior_variance
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

        # initial prediction model as guided condition
        if config.diffusion.apply_aux_cls:
            self.cond_pred_model = AuxCls(config).to(self.device)
            self.aux_cost_function = nn.MSELoss() #nn.CrossEntropyLoss()
        else:
            pass

        # scaling temperature for NLL and ECE computation
        self.tuned_scale_T = None
        self.std = config.data.std     

    # Compute guiding prediction as diffusion condition
    def compute_guiding_prediction(self, x):
        """
        Compute y_0_hat, to be used as the Gaussian mean at time step T.
        """
        if self.config.model.arch == "simple" or \
                (self.config.model.arch == "linear" and self.config.data.dataset == "MNIST"):
            x = torch.flatten(x, 1)
        y_pred, y_global, y_local = self.cond_pred_model(x)
        return y_pred, y_global, y_local

    def evaluate_guidance_model(self, dataset_loader, train=False):
        """
        Evaluate guidance model by reporting train or test set accuracy.
        """
        y_acc_list = []
        for step, feature_label_set in tqdm(enumerate(dataset_loader)):
            x_batch, y_labels_batch, img_paths = feature_label_set
            y_labels_batch = y_labels_batch.reshape(-1, 1)
            y_pred_prob,_,_ = self.compute_guiding_prediction(
                x_batch.to(self.device))  # (batch_size, n_classes)
            y_pred_label = y_pred_prob.cpu().detach().numpy()  
            y_labels_batch = y_labels_batch.cpu().detach().numpy()
           
            #compute mae
            mae = np.abs(y_pred_label - y_labels_batch)
            y_acc = mae 
            if len(y_acc_list) == 0:
                y_acc_list = y_acc
            else:
                y_acc_list = np.concatenate([y_acc_list, y_acc], axis=0)
        y_acc_all = np.mean(y_acc_list)
        return y_acc_all

    def nonlinear_guidance_model_train_step(self, x_batch, y_batch, aux_optimizer):
        """
        One optimization step of the non-linear guidance model that predicts y_0_hat.
        """
        y_batch_pred,y_global,y_local = self.compute_guiding_prediction(x_batch)
        y_batch = y_batch.unsqueeze(1)
        aux_cost = self.aux_cost_function(y_batch_pred.float(), y_batch.float())
        
        # update non-linear guidance model
        aux_optimizer.zero_grad()
        aux_cost.backward()
        aux_optimizer.step()
        return aux_cost.cpu().item()

    def train(self):
        # start a new wandb run to track this script
        wandb.init(
                # set the wandb project where this run will be logged
                project="diff",
            )
        args = self.args
        config = self.config
        tb_logger = self.config.tb_logger
        data_object, train_dataset, test_dataset = get_dataset(args, config)
        print('loading dataset..')

        train_loader = data.DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            #sampler=sampler
        )
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=config.testing.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        )
        print('successfully load')
        model = ConditionalModel(config, guidance=config.diffusion.include_guidance)
        model = model.to(self.device)
        y_acc_aux_model = self.evaluate_guidance_model(test_loader, train=False)
        logging.info("\nBefore training, the guidance regression accuracy on the test set is {:.8f}.\n\n".format(
            y_acc_aux_model))

        optimizer = get_optimizer(self.config.optim, model.parameters())
        criterion = nn.MSELoss() 

        # apply an auxiliary optimizer for the guidance regression
        if config.diffusion.apply_aux_cls:
            aux_optimizer = get_optimizer(self.config.aux_optim,
                                          self.cond_pred_model.parameters())

        if self.config.model.ema:
            ema_helper = EMA(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        if config.diffusion.apply_aux_cls:
            early_stopper = EarlyStopping(patience=config.diffusion.nonlinear_guidance.patience,
                                              delta=config.diffusion.nonlinear_guidance.delta)
            
            if hasattr(config.diffusion, "trained_aux_cls_ckpt_path"):  # load saved auxiliary regression
                aux_states = torch.load(os.path.join(config.diffusion.trained_aux_cls_ckpt_path,
                                                     config.diffusion.trained_aux_cls_ckpt_name),
                                        map_location=self.device)
                self.cond_pred_model.load_state_dict(aux_states['state_dict'], strict=True)
                self.cond_pred_model.eval()
            elif hasattr(config.diffusion, "trained_aux_cls_log_path"):
                aux_states = torch.load(os.path.join(config.diffusion.trained_aux_cls_log_path, "aux_ckpt.pth"),
                                        map_location=self.device)
                self.cond_pred_model.load_state_dict(aux_states[0], strict=True)
                self.cond_pred_model.eval()
            else:  # pre-train the guidance auxiliary regression
                assert config.diffusion.aux_cls.pre_train
                self.cond_pred_model.train()
                pretrain_start_time = time.time()
                for epoch in range(config.diffusion.aux_cls.n_pretrain_epochs):
                    for feature_label_set in train_loader:
                        x_batch, y_one_hot_batch, img_paths = feature_label_set
                        aux_loss = self.nonlinear_guidance_model_train_step(x_batch.to(self.device),
                                                                            y_one_hot_batch.to(self.device),
                                                                            aux_optimizer)
                        
                        early_stopper(val_cost=aux_loss, epoch=epoch)
                        
                        if early_stopper.early_stop:
                            print(("Obtained best performance on validation set after Epoch {}; " +
                                        "early stopping at Epoch {}.").format( early_stopper.best_epoch, epoch))
                            break
                    if epoch % config.diffusion.aux_cls.logging_interval == 0:
                        logging.info(
                            f"epoch: {epoch}, guidance auxiliary regression pre-training loss: {aux_loss}"
                        )
                pretrain_end_time = time.time()
                logging.info("\nPre-training of guidance auxiliary regression took {:.4f} minutes.\n".format(
                    (pretrain_end_time - pretrain_start_time) / 60))
                # save auxiliary model after pre-training
                aux_states = [
                    self.cond_pred_model.state_dict(),
                    aux_optimizer.state_dict(),
                ]
                torch.save(aux_states, os.path.join(self.args.log_path, "aux_ckpt.pth"))
            # report accuracy on both training and test set for the pre-trained auxiliary regression
            y_acc_aux_model = self.evaluate_guidance_model(train_loader,train=True)
            logging.info("\nAfter pre-training, guidance regression accuracy on the training set is {:.8f}.".format(
                y_acc_aux_model))
            y_acc_aux_model = self.evaluate_guidance_model(test_loader, train=False)
            logging.info("\nAfter pre-training, guidance regression accuracy on the val set is {:.8f}.\n".format(
                y_acc_aux_model))
            

        if not self.args.train_guidance_only:
            start_epoch, step = 0, 0
            early_stopper_main = EarlyStopping(patience=config.diffusion.nonlinear_guidance.patience,
                                              delta=config.diffusion.nonlinear_guidance.delta)
            if self.args.resume_training:
                states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"),
                                    map_location=self.device)
                model.load_state_dict(states[0])

                states[1]["param_groups"][0]["eps"] = self.config.optim.eps
                optimizer.load_state_dict(states[1])
                start_epoch = states[2]
                step = states[3]
                if self.config.model.ema:
                    ema_helper.load_state_dict(states[4])
                # load auxiliary model
                if config.diffusion.apply_aux_cls and (
                        hasattr(config.diffusion, "trained_aux_cls_ckpt_path") is False) and (
                        hasattr(config.diffusion, "trained_aux_cls_log_path") is False):
                    aux_states = torch.load(os.path.join(self.args.log_path, "aux_ckpt.pth"),
                                            map_location=self.device)
                    self.cond_pred_model.load_state_dict(aux_states[0])
                    aux_optimizer.load_state_dict(aux_states[1])

            max_accuracy = np.inf

            
            if config.diffusion.noise_prior:  # apply 0 instead of f_phi(x) as prior mean
                logging.info("Prior distribution at timestep T has a mean of 0.")
            if args.add_ce_loss:
                logging.info("Apply MSE as an auxiliary loss during training.")
            for epoch in range(start_epoch, self.config.training.n_epochs):
                data_start = time.time()
                data_time = 0
                for i, feature_label_set in enumerate(train_loader):
                    x_batch, y_one_hot_batch ,paths = feature_label_set # (images, target, path)
                    if config.optim.lr_schedule:
                        adjust_learning_rate(optimizer, i / len(train_loader) + epoch, config)
                    n = x_batch.size(0)
                    # record unflattened x as input to guidance aux regression
                    x_unflat_batch = x_batch.to(self.device)
                    if config.data.dataset == "toy" or config.model.arch in ["simple", "linear"]:
                        x_batch = torch.flatten(x_batch, 1)
                    data_time += time.time() - data_start
                    model.train()
                    self.cond_pred_model.eval()
                    step += 1

                    # antithetic sampling
                    t = torch.randint(
                        low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                    ).to(self.device)
                    t = torch.cat([t, self.num_timesteps - 1 - t], dim=0)[:n]

                    # noise estimation loss
                    x_batch = x_batch.to(self.device)
                    y_0_hat_batch, y_0_global, y_0_local = self.compute_guiding_prediction(x_unflat_batch)
                    y_T_mean = y_0_hat_batch
                    if config.diffusion.noise_prior:  # apply 0 instead of f_phi(x) as prior mean
                        y_T_mean = torch.zeros(y_0_hat_batch.shape).to(y_0_hat_batch.device)
                    y_0_batch = y_one_hot_batch.float().to(self.device).squeeze()
                    e = torch.randn_like(y_0_batch).to(y_0_batch.device)
                    
                    y_t_batch = q_sample(y_0_batch, y_T_mean,
                                         self.alphas_bar_sqrt, self.one_minus_alphas_bar_sqrt, t, noise=e)
                    y_t_batch_global = q_sample(y_0_batch, y_0_global,
                                        self.alphas_bar_sqrt, self.one_minus_alphas_bar_sqrt, t, noise=e)
                    y_t_batch_local = q_sample(y_0_batch, y_0_local,
                                        self.alphas_bar_sqrt, self.one_minus_alphas_bar_sqrt, t, noise=e)
                    output = model(x_batch, y_t_batch, t, y_0_hat_batch)
                    output_global = model(x_batch, y_t_batch_global, t, y_0_global)
                    output_local = model(x_batch, y_t_batch_local, t, y_0_local)

                    loss = (e.flatten() - output.flatten()).square().mean() + 0.5*(compute_mmd(e,output_global) + compute_mmd(e,output_local))  # use the same noise sample e during training to compute loss
                    
                    if not tb_logger is None:
                        tb_logger.add_scalar("loss", loss, global_step=step)

                    if step % self.config.training.logging_freq == 0 or step == 1:
                        logging.info(
                            (
                                    f"epoch: {epoch}, step: {step} "
                                    f"Noise Estimation loss: {loss.item()}, " +
                                    f"data time: {data_time / (i + 1)}"
                            )
                        )
                        wandb.log({"MSE loss train": loss.item(), "epoch": epoch})

                    # optimize diffusion model that predicts eps_theta
                    optimizer.zero_grad()
                    loss.backward()
                    try:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.optim.grad_clip
                        )
                    except Exception:
                        pass
                    optimizer.step()
                    if self.config.model.ema:
                        ema_helper.update(model)

                    # joint train aux regression along with diffusion model
                    if config.diffusion.apply_aux_cls and config.diffusion.aux_cls.joint_train:
                        self.cond_pred_model.train()
                        aux_loss = self.nonlinear_guidance_model_train_step(x_unflat_batch, y_0_batch,
                                                                            aux_optimizer)
                        if step % self.config.training.logging_freq == 0 or step == 1:
                            logging.info(
                                f"meanwhile, guidance auxiliary regression joint-training loss: {aux_loss}"
                            )

                    # save diffusion model
                    if step % self.config.training.snapshot_freq == 0 or step == 1:
                        states = [
                            model.state_dict(),
                            optimizer.state_dict(),
                            epoch,
                            step,
                        ]
                        if self.config.model.ema:
                            states.append(ema_helper.state_dict())

                        if step > 1:  # skip saving the initial ckpt
                            torch.save(
                                states,
                                os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                            )
                        # save current states
                        torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                        # save auxiliary model
                        if config.diffusion.apply_aux_cls and config.diffusion.aux_cls.joint_train:
                            aux_states = [
                                self.cond_pred_model.state_dict(),
                                aux_optimizer.state_dict(),
                            ]
                            if step > 1:  # skip saving the initial ckpt
                                torch.save(
                                    aux_states,
                                    os.path.join(self.args.log_path, "aux_ckpt_{}.pth".format(step)),
                                )
                            torch.save(aux_states, os.path.join(self.args.log_path, "aux_ckpt.pth"))

                    data_start = time.time()

                logging.info(
                    (f"epoch: {epoch}, step: {step}, Noise Estimation loss: {loss.item()}, " +
                     f"data time: {data_time / (i + 1)}")
                )

                # Evaluate
                if epoch % self.config.training.validation_freq == 0 \
                        or epoch + 1 == self.config.training.n_epochs:
                        model.eval()
                        self.cond_pred_model.eval()
                        acc_avg = 0.
                        kappa_avg = 0.
                        y1_true=None
                        y1_pred=None
                        paths=None
                        for test_batch_idx, (images, target, path) in enumerate(test_loader):
                            images_unflat = images.to(self.device)
                            if config.data.dataset == "toy" \
                                    or config.model.arch == "simple" \
                                    or config.model.arch == "linear":
                                images = torch.flatten(images, 1)
                            images = images.to(self.device)
                            target = target.to(self.device)
                            
                            # target_vec = nn.functional.one_hot(target).float().to(self.device)
                            with torch.no_grad():
                                target_pred, y_global, y_local = self.compute_guiding_prediction(images_unflat)
                                # prior mean at timestep T
                                y_T_mean = target_pred
                                if config.diffusion.noise_prior:  # apply 0 instead of f_phi(x) as prior mean
                                    y_T_mean = torch.zeros(target_pred.shape).to(target_pred.device)
                                if not config.diffusion.noise_prior:  # apply f_phi(x) instead of 0 as prior mean
                                    target_pred, y_global, y_local = self.compute_guiding_prediction(images_unflat)
                                    #target_pred = target_pred.softmax(dim=1)

                                label_t_0 = p_sample_loop(model, images, target_pred, y_T_mean,
                                                          self.num_timesteps, self.alphas,
                                                          self.one_minus_alphas_bar_sqrt,
                                                          only_last_sample=True)                               
                                y1_pred = torch.cat([y1_pred, label_t_0]) if y1_pred is not None else label_t_0
                                y1_true = torch.cat([y1_true, target]) if y1_true is not None else target 
                                paths = np.concatenate([paths, path]) if paths is not None else path       
                                
                    
                        #compute mae
                        mae_avg = compute_mae_score(y1_true,y1_pred,paths,self.std)#.item()
                        #compute mae
                        #acc_avg = np.average(np.abs(y1_true.detach().cpu().numpy() - y1_pred.cpu().numpy().flatten()))     
                          
                        #acc_avg=mae_avg
                        if mae_avg < max_accuracy:
                            logging.info("Update min MAE at Epoch {}.".format(epoch))
                            states = [
                                model.state_dict(),
                                optimizer.state_dict(),
                                epoch,
                                step,
                            ]
                            torch.save(states, os.path.join(self.args.log_path, "ckpt_best.pth"))
                            aux_states = [
                                    self.cond_pred_model.state_dict(),
                                    aux_optimizer.state_dict(),
                                ]
                            torch.save(aux_states, os.path.join(self.args.log_path, "aux_ckpt_best.pth"))
                        # check if mae_avg is not nan
                        if mae_avg is not np.nan:
                            max_accuracy = min(max_accuracy, mae_avg)
                        if not tb_logger is None:
                            tb_logger.add_scalar('MAE', mae_avg, global_step=step)
                        logging.info(
                            (
                                    f"epoch: {epoch}, step: {step}, " +
                                    f"MAE: {mae_avg},"
                                    f"Min MAE: {max_accuracy:.2f}"
                            )
                        )
                        
                        wandb.log({"MSE loss val": mae_avg, "epoch": epoch})
                        early_stopper_main(val_cost=mae_avg, epoch=epoch)
                        if early_stopper_main.early_stop:
                            print(("Obtained best performance on validation set after Epoch {}; " +
                               "early stopping at Epoch {}.").format(early_stopper_main.best_epoch, epoch))
                            break               

            # save the model after training is finished
            states = [
                model.state_dict(),
                optimizer.state_dict(),
                epoch,
                step,
            ]
            if self.config.model.ema:
                states.append(ema_helper.state_dict())
            torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))
            # save auxiliary model after training is finished
            if config.diffusion.apply_aux_cls and config.diffusion.aux_cls.joint_train:
                aux_states = [
                    self.cond_pred_model.state_dict(),
                    aux_optimizer.state_dict(),
                ]
                torch.save(aux_states, os.path.join(self.args.log_path, "aux_ckpt.pth"))
                # report training set accuracy if applied joint training
                y_acc_aux_model = self.evaluate_guidance_model(train_loader,train=True)
                logging.info("After joint-training, guidance MAE on the training set is {:.8f}.".format(
                    y_acc_aux_model))
                # report test set accuracy if applied joint training
                y_acc_aux_model = self.evaluate_guidance_model(test_loader,train=False)
                logging.info("After joint-training, guidance MAE on the test set is {:.8f}.".format(
                    y_acc_aux_model))

    def test(self):
        args = self.args
        config = self.config
        data_object, train_dataset, test_dataset = get_dataset(args, config)
        log_path = os.path.join(self.args.log_path)

        test_loader = data.DataLoader(
            test_dataset,
            batch_size=config.testing.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        )
        model = ConditionalModel(config, guidance=config.diffusion.include_guidance)
        
        if getattr(self.config.testing, "ckpt_id", None) is None:
            if args.eval_path is not None:
                states = torch.load(args.eval_path, map_location=self.device)
                #logging.info(f"Loading from:", args.eval_path)
            elif args.eval_best:
                ckpt_id = 'best'
                states = torch.load(os.path.join(log_path, f"ckpt_{ckpt_id}.pth"),
                                    map_location=self.device)
                logging.info(f"Loading from: {log_path}/ckpt_{ckpt_id}.pth")
            else:
                ckpt_id = 'last'
                states = torch.load(os.path.join(log_path, "ckpt.pth"),
                                    map_location=self.device)
                logging.info(f"Loading from: {log_path}/ckpt_{ckpt_id}.pth")
        else:
            states = torch.load(os.path.join(log_path, f"ckpt_{self.config.testing.ckpt_id}.pth"),
                                map_location=self.device)
            ckpt_id = self.config.testing.ckpt_id
            logging.info(f"Loading from: {log_path}/ckpt_{ckpt_id}.pth")
        model = model.to(self.device)
        model.load_state_dict(states[0], strict=True)

        num_params = 0
        for param in model.parameters():
            num_params += param.numel()

        # load auxiliary model
        if config.diffusion.apply_aux_cls:
            if hasattr(config.diffusion, "trained_aux_cls_ckpt_path"):
                aux_states = torch.load(os.path.join(config.diffusion.trained_aux_cls_ckpt_path,
                                                     config.diffusion.trained_aux_cls_ckpt_name),
                                        map_location=self.device)
                self.cond_pred_model.load_state_dict(aux_states['state_dict'], strict=True)
            else:
                aux_cls_path = log_path
                
                if hasattr(config.diffusion, "trained_aux_cls_log_path"):
                    aux_cls_path = config.diffusion.trained_aux_cls_log_path
                    aux_states = torch.load(os.path.join(aux_cls_path, "aux_ckpt_best.pth"),
                                        map_location=self.device)
                    logging.info(f"Loading from: {aux_cls_path}/aux_ckpt_best.pth")
                elif args.eval_path_aux is not None:
                    print("Loading from:", args.eval_path_aux)
                    aux_states = torch.load(args.eval_path_aux, map_location=self.device)
                else: 
                    aux_cls_path = log_path
                    aux_states = torch.load(os.path.join(aux_cls_path, "aux_ckpt_best.pth"),
                                        map_location=self.device)
                    logging.info(f"Loading from: {aux_cls_path}/aux_ckpt_best.pth")
                
                self.cond_pred_model.load_state_dict(aux_states[0], strict=False)

        # Evaluate
        model.eval()
        self.cond_pred_model.eval()
        
        y1_true = None
        y1_pred = None
        paths = None
        for test_batch_idx, (images, target, path) in enumerate(test_loader):
            images_unflat = images.to(self.device)
            target = target.to(self.device)
            target_vec = target.float().to(self.device)
            with torch.no_grad():
                target_pred, y_global, y_local = self.compute_guiding_prediction(images_unflat)
                
                # prior mean at timestep T
                y_T_mean = target_pred
                if config.diffusion.noise_prior:  # apply 0 instead of f_phi(x) as prior mean
                    y_T_mean = torch.zeros(target_pred.shape).to(target_pred.device)
                if not config.diffusion.noise_prior:  # apply f_phi(x) instead of 0 as prior mean
                    target_pred, y_global, y_local = self.compute_guiding_prediction(images_unflat)
          
                label_t_0 = p_sample_loop(model, images_unflat, target_pred, y_T_mean,
                                                self.test_num_timesteps, self.alphas,
                                                self.one_minus_alphas_bar_sqrt,
                                                only_last_sample=True)
               
                y1_pred = torch.cat([y1_pred, target_pred]) if y1_pred is not None else target_pred
                y1_true = torch.cat([y1_true, target]) if y1_true is not None else target      
                paths = np.concatenate([paths, path]) if paths is not None else path             

        mae_avg = compute_mae_score(y1_true, y1_pred, paths,self.std)
        
