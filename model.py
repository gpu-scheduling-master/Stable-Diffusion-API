from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained(
    "sd-legacy/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)

pipe.to("cuda")


def gen_image(prompt: str):
    return {"image": pipe(prompt=prompt).images[0]}
