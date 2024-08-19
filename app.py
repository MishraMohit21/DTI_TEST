from flask import Flask, request, render_template, redirect, url_for, send_file, jsonify
import os
import io
from werkzeug.utils import secure_filename
import cv2
import script  # Import your script

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            try:
                result_img, psnr, mse = script.process_image_from_path(file_path, population_size=10, generations=50)
                result_img_path = os.path.join(RESULTS_FOLDER, 'result_image.png')
                cv2.imwrite(result_img_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))

                return render_template('result.html', 
                                       result_image_url=url_for('get_result_image', filename='result_image.png'),
                                       psnr=psnr, mse=mse)
            except Exception as e:
                return jsonify({"error": str(e)}), 500
    return render_template('index.html')

@app.route('/results/<filename>')
def get_result_image(filename):
    return send_file(os.path.join(app.config['RESULTS_FOLDER'], filename))

if __name__ == "__main__":
    app.run(debug=True)
