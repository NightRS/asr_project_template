FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-devel

RUN conda install -y \
    editdistance==0.6.1 \
    datasets==2.12.0 \
    librosa==0.9.2 \
    matplotlib==3.8.0 \
    pandas==2.1.1 \
    pyctcdecode==0.5.0 \
    pysoundfile==0.11.0 \
    tensorboard==2.12.1 \
    wandb==0.15.12 \
    -c pytorch \
    -c nvidia \
    -c defaults \
    -c conda-forge \
    --solver=libmamba \
    --freeze-installed

RUN pip install torch_audiomentations==0.11.0
