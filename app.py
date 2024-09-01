from fastapi import FastAPI
from model import gen_image
app = FastAPI(title="Stable-Diffusion API", debug=True)


@app.get("/healthcheck")
async def healthcheck():
    return {"status": "ok"}


@app.post("/gen-img")
async def gen_img(prompt: str):
    return gen_image(prompt)