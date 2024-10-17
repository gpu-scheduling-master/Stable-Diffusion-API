FROM docker.io/pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
ENV DEBIAN_FRONTEND=noninteractive PIP_PREFER_BINARY=1

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && apt-get clean

COPY ./requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "app:fastapi_app", "--host", "0.0.0.0", "--port", "8000"]