
from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import torch
import pywt
from ultralytics import YOLO
from PIL import Image
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Load YOLO models
detection_model = YOLO(r'D:\NCKH\RECAPTURE_SCREENSHOST_IMAGE\Website\detect\weights\best.pt')
classification_model = YOLO(r'D:\NCKH\RECAPTURE_SCREENSHOST_IMAGE\Website\Screen\weights\best.pt')
printed_classification_model = YOLO(r'D:\NCKH\RECAPTURE_SCREENSHOST_IMAGE\Website\Print\weights\best.pt')

# Set models to evaluation mode
detection_model.eval()
classification_model.eval()
printed_classification_model.eval()

def detect_and_crop(image):
    """Detect and crop ID Card from the image."""
    try:
        results = detection_model(image)
        boxes = results[0].boxes.data
        if len(boxes) == 0:
            return None, "Không phát hiện được vùng ID Card."
        x_min, y_min, x_max, y_max = map(int, boxes[0][:4].cpu().numpy())
        if x_max <= x_min or y_max <= y_min:
            return None, "Lỗi: Hộp nhận diện không hợp lệ."
        image_cv = np.array(image)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
        cropped_image = image_cv[y_min:y_max, x_min:x_max]
        if cropped_image.size == 0 or cropped_image.shape[0] == 0 or cropped_image.shape[1] == 0:
            return None, "Lỗi: Ảnh cắt trống."
        return cropped_image, None
    except Exception as e:
        return None, f"Lỗi khi cắt ảnh: {str(e)}"

def process_image_advanced(image):
    """Preprocess image for screen capture classification."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.log(1 + np.abs(fshift))
        magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_combined = cv2.magnitude(sobel_x, sobel_y)
        sobel_normalized = cv2.convertScaleAbs(sobel_combined)
        coeffs = pywt.dwt2(gray, 'haar')
        cA, (cH, cV, cD) = coeffs
        wavelet_combined = cv2.convertScaleAbs(cA)
        wavelet_combined = cv2.resize(wavelet_combined, (gray.shape[1], gray.shape[0]))
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        combined = cv2.addWeighted(magnitude_spectrum, 0.3, sobel_normalized, 0.3, 0)
        combined = cv2.addWeighted(combined, 0.5, wavelet_combined, 0.3, 0)
        combined = cv2.addWeighted(combined, 0.8, blurred, 0.2, 0)
        return combined
    except Exception as e:
        raise Exception(f"Lỗi khi xử lý ảnh nâng cao: {str(e)}")

def build_gabor_filters(ksize=31, sigma=4.0, theta=np.pi/3, lambd=4.3, gamma=0.005):
    filters = []
    kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
    filters.append(kern)
    return filters

def apply_gabor_filters(image, filters):
    try:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.equalizeHist(gray_image)
        gray_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
        gray_image = cv2.bilateralFilter(gray_image, 7, 100, 100)
        responses = [cv2.filter2D(gray_image, cv2.CV_8UC1, kern) for kern in filters]
        return np.maximum.reduce(responses)
    except Exception as e:
        raise Exception(f"Lỗi khi áp dụng bộ lọc Gabor: {str(e)}")

def process_image_simple(cropped_image):
    filters = build_gabor_filters()
    processed_image = apply_gabor_filters(cropped_image, filters)
    return processed_image

@app.route('/')
def index():
    try:
        print("Serving website.html")
        if not os.path.exists('website.html'):
            print("Error: website.html not found")
            return jsonify({'error': 'website.html not found'}), 404
        return send_file('website.html')
    except Exception as e:
        print(f"Error serving website.html: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/favicon.ico')
def favicon():
    return '', 204  # Ignore favicon request for now

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        print("Error: No image file in request")
        return jsonify({'error': 'Không có file ảnh'}), 400
    file = request.files['image']
    try:
        print("Received image:", file.filename)
        image = Image.open(file.stream).convert('RGB')
        cropped_image, error = detect_and_crop(image)
        if error:
            print("Crop error:", error)
            return jsonify({'error': error}), 400
        processed_screen = process_image_advanced(cropped_image)
        processed_printed = process_image_simple(cropped_image)
        cv2.imwrite('processed_screen.jpg', processed_screen)
        cv2.imwrite('processed_printed.jpg', processed_printed)
        processed_screen_pil = Image.fromarray(cv2.cvtColor(processed_screen, cv2.COLOR_GRAY2RGB)).convert('RGB')
        processed_printed_pil = Image.fromarray(processed_printed).convert('RGB')
        results_screen = classification_model(processed_screen_pil)
        predictions_screen = results_screen[0].probs
        class_mapping = {0: "FAKE", 1: "REAL"}
        result_screen = class_mapping.get(np.argmax(predictions_screen.data.cpu().numpy()), "UNKNOWN")
        print("Screen result:", result_screen)
        results_print = printed_classification_model(processed_printed_pil)
        predictions_print = results_print[0].probs
        result_print = class_mapping.get(np.argmax(predictions_print.data.cpu().numpy()), "UNKNOWN")
        print("Print result:", result_print)
        final_result = "FAKE" if result_screen == "FAKE" or result_print == "FAKE" else "REAL"
        print("Final result:", final_result)
        return jsonify({
            'result': final_result,
            'processed_screen': 'processed_screen.jpg',
            'processed_printed': 'processed_printed.jpg'
        })
    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/processed_image/<image_type>', methods=['GET'])
def get_processed_image(image_type):
    try:
        if image_type == 'screen':
            filename = 'processed_screen.jpg'
        elif image_type == 'printed':
            filename = 'processed_printed.jpg'
        else:
            return jsonify({'error': 'Loại ảnh không hợp lệ'}), 400
        if not os.path.exists(filename):
            print(f"Error: {filename} not found")
            return jsonify({'error': f'{filename} not found'}), 404
        return send_file(filename, mimetype='image/jpeg')
    except Exception as e:
        print(f"Error serving {image_type} image: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Server is starting...")
    app.run(host='0.0.0.0', port=5000, debug=True)