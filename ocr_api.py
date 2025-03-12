from flask import Flask, request, jsonify
# from flask_cors import CORS
import numpy as np
import onnxruntime as ort
from PIL import Image
import io
import string

app = Flask(__name__)
# CORS(app)  # Enable CORS

# Constant
MODEL_PATH = "model.onnx"
IMAGE_SIZE = (120, 100)
CHARS = string.ascii_lowercase # from 'a' to 'z'

# Load ONNX model
ort_session = ort.InferenceSession(MODEL_PATH)
    
def normalize_image_to_np_array(file, size):
    image = Image.open(io.BytesIO(file.read()))
    image = image.convert("L")  # gray style
    image = image.point(lambda p: 255 if p > 128 else 0) # binarization
    image = image.resize(size)  # unify size
    image_array = np.array(image) / 255.0  # normalization
    return image_array

def predict_captcha(file):
    # import
    image_array = normalize_image_to_np_array(file, IMAGE_SIZE)
    image_array = image_array.reshape(1, 1, IMAGE_SIZE[1], IMAGE_SIZE[0])  # (batch_size, channels, height, width)
    image_array = image_array.astype(np.float32)

    # predict
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    result = ort_session.run([output_name], {input_name: image_array})

    # decode
    predicted_indices = np.argmax(result[0], axis=2).flatten()
    predicted_text = ''.join(CHARS[i] for i in predicted_indices)
    
    return predicted_text

@app.route("/recognize", methods=["POST"])
def recognize():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    predicted_text = predict_captcha(file)
    return jsonify({"text": predicted_text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
