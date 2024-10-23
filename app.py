from fastapi import FastAPI
from fastapi.responses import Response
from model import gen_image
app = FastAPI(title="Stable-Diffusion API", debug=True)


@app.get("/healthcheck")
async def healthcheck():
    return {"status": "ok"}


@app.post("/gen-img")
async def gen_img(prompt: str, step: None | int):
    return Response(content=gen_image(prompt, step), media_type="image/png")