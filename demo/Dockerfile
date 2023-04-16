FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt requirements.txt

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y  --no-install-recommends && apt-get clean &&\
    rm -rf /var/lib/apt/lists/* 
RUN pip install -r requirements.txt &&\
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]