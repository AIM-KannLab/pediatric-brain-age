FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime
#FROM puzlcloud/pytorch:1.10.1-cuda11.3-cudnn8-jupyter-g1-1.1.0-python3.8

USER 0
#RUN python3.8 -m pip install --upgrade pip 
ENV TZ=US/Eastern
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt update && apt install -y tzdata

RUN apt update && \
    apt install --no-install-recommends -y build-essential software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install --no-install-recommends -y python3.8 python3-pip python3-setuptools python3-distutils && \
    apt-get install -y wget && \
    apt clean && rm -rf /var/lib/apt/lists/*
RUN python3.8 -m pip install --upgrade pip 

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH
COPY environment.yml /environment.yml 

#RUN conda update -n base -c defaults conda
RUN python3 -m pip install --upgrade pip
RUN pip install statsmodels==0.12.1 torchaudio==0.10.0 torchvision==0.11.1
RUN pip install simpleitk itk-elastix imageio matplotlib opencv-python pandas pyyaml \
    scikit-image scikit-learn scipy tensorboard tqdm seaborn nibabel

COPY configs/ /configs/
COPY run.py /run.py
COPY HDBET/ /HDBET/
COPY dataloader/ /dataloader/
COPY dataset/ /dataset/ 
COPY example_data/ /example_data/

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /
ENV MKL_SERVICE_FORCE_INTEL=1
ENTRYPOINT ["python", "/run.py"]



