from diffusers import DiffusionPipeline
import torch
import io

pipe = DiffusionPipeline.from_pretrained(
    "sd-legacy/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)

pipe.to("cuda")


def gen_image(prompt: str):
    img = pipe(prompt=prompt).images[0]
    byte_arr = io.BytesIO()
    img.save(byte_arr, format="PNG")
    return byte_arr.getvalue()
