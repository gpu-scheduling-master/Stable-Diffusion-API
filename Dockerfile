FROM docker.io/pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && apt-get clean

COPY ./requirements.txt .

RUN pip install -r requirements.txt

RUN python -c "from diffusers import StableDiffusionPipeline; StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', use_safetensors=True).save_pretrained('/models/stable-diffusion-v1-4')"

COPY . .

ENV SD_MODEL_PATH="/models/stable-diffusion-v1-4"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]