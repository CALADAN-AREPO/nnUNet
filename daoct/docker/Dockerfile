FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

RUN pip install --no-cache-dir \
    nnunetv2 \
    "transformers>=4.45.0" \
    peft \
    "sam3 @ git+https://github.com/facebookresearch/sam3.git" \
    timm \
    einops \
    scikit-image \
    connected-components-3d \
    huggingface_hub

RUN python -c "from huggingface_hub import snapshot_download; \
               snapshot_download('lal-Joey/MedSAM3_v1', local_dir='/weights/medsam3_lora', token=os.environ.get('HF_TOKEN', None))"

COPY . /workspace/
WORKDIR /workspace

RUN pip install -e .

ENV nnUNet_raw=/data/raw
ENV nnUNet_preprocessed=/data/preprocessed
ENV nnUNet_results=/data/results

ENTRYPOINT ["python", "daoct/docker/train.py"]
