
from flask import Flask, request, redirect, url_for, send_from_directory, render_template
import os
from werkzeug.utils import secure_filename
import subprocess
import sys

UPLOAD_FOLDER = 'uploads'
DETECT_FOLDER = 'test_images2'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DETECT_FOLDER'] = DETECT_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clear_detect_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error: {e}")

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', message='No selected file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Clear the DETECT_FOLDER before processing the new image
            if os.path.exists(app.config['DETECT_FOLDER']):
                clear_detect_folder(app.config['DETECT_FOLDER'])
            else:
                os.makedirs(app.config['DETECT_FOLDER'])

            # Call main.py with the uploaded file path
            subprocess.run([sys.executable, 'main.py', '--data', file_path])
            
            return redirect(url_for('uploaded_file', filename=filename))
    return render_template('index.html', message='')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    if not os.path.exists(DETECT_FOLDER):
        os.makedirs(DETECT_FOLDER)
    app.run(debug=True)
