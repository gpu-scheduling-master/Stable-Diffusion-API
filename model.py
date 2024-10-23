from diffusers import DiffusionPipeline
import torch
import os
import io

pipe = DiffusionPipeline.from_pretrained(
    os.environ["SD_MODEL"],
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)

pipe.to("cuda")


def gen_image(prompt: str, steps: int = 25):
    img = pipe(prompt=prompt, num_inference_steps=steps).images[0]
    byte_arr = io.BytesIO()
    img.save(byte_arr, format="PNG")
    return byte_arr.getvalue()
