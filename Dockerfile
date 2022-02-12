ARG PYTORCH="1.6.0"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TZ=Asia/Ho_Chi_Minh
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV FORCE_CUDA="1"
ENV LANG=C.UTF-8

# Setup basic requirements
RUN apt-get update && apt-get upgrade -y
RUN apt-get update && apt-get install -y libgbm-dev -y \
    software-properties-common dirmngr -y \
    build-essential -y \
    libgl1-mesa-glx libxrender1 libfontconfig1 -y \
    libglib2.0-0 -y \
    libsm6 libxext6 libxrender-dev -y \
    gnupg2 -y \
    zip -y \
    git -y ninja-build -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN conda clean --all
RUN pip install mmcv-full===1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html
RUN pip install mmdet==2.14.0

WORKDIR /app
COPY ./app /app
COPY ./download_weights.sh /app

# Setup for VietOCR
RUN pip install scikit-build easydict \
    && pip install vietocr==0.3.5

# Download model weights
RUN mkdir weights && pip install gdown && bash download_weights.sh

# Setup for FastAPI
RUN pip install fastapi uvicorn[standard] python-multipart aiofiles

WORKDIR /app/libs/mmocr
RUN pip uninstall -y opencv-python && pip install -r requirements.txt
WORKDIR /app

EXPOSE 80

# CMD ["uvicorn", "main:app", "--workers", "1", "--host", "0.0.0.0", "--port", "80"]
