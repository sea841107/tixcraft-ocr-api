FROM python:3.10-slim

WORKDIR /app

COPY . /app

COPY model.onnx /app/model.onnx

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python", "ocr_api.py"]
