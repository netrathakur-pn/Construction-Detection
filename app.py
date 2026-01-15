from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
from detection import detect_changes

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_images():
    before = request.files['before']
    after = request.files['after']

    if not before or not after:
        return redirect(url_for('index'))

    before_path = os.path.join(UPLOAD_FOLDER, 'before.jpg')
    after_path = os.path.join(UPLOAD_FOLDER, 'after.jpg')

    before.save(before_path)
    after.save(after_path)

    # Resize both images to same dimensions
    img1 = cv2.imread(before_path)
    img2 = cv2.imread(after_path)

    if img1 is None or img2 is None:
        return "❌ Error: One of the images could not be read properly."

    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])

    img1 = cv2.resize(img1, (w, h))
    img2 = cv2.resize(img2, (w, h))

    cv2.imwrite(before_path, img1)
    cv2.imwrite(after_path, img2)

    # Detect changes
   # Detect changes
    result_path = os.path.join(RESULT_FOLDER, 'result.jpg')
    result_img, metrics = detect_changes(before_path, after_path, result_path)

    return render_template(
    'result.html',
    before_img=before_path,
    after_img=after_path,
    result_img=result_path,
    mask_img=metrics["mask_path"],
    detected_count=metrics["detected_changes"],
    accuracy=metrics["change_ratio_percent"],
    efficiency=metrics["confidence_score"]
)


if __name__ == '__main__':
    app.run(debug=True)
