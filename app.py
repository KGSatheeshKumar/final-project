from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import csv
import subprocess

app = Flask(__name__)

# Ensure 'data' folder exists
if not os.path.exists('data'):
    os.makedirs('data')

@app.route('/')
def index():
    return render_template('index.html')

from flask import send_from_directory
@app.route('/send_image/<filename>')
def send_image(filename):
    return send_from_directory('data', filename)

@app.route('/data_gathering', methods=['POST'])
def data_gathering():
    user_id = request.form['user_id']
    name = request.form['name']
    crime = request.form['crime']
    
    # Save name and crime to a CSV file
    with open('data/criminal_info.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([user_id, name, crime])

    # Start capturing images from the webcam
    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    count = 0

    while True:
        _, img = cam.read()
        img = cv2.flip(img, 1)  # Flip camera vertically
        faces = detector.detectMultiScale(img, 1.3, 5)
        
        for (x, y, w, h) in faces:
            count += 1
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            # Save the captured image with user_id, name, and count
            cv2.imwrite(f"data/{user_id}_{name}_{count}.jpg", img[y:y + h, x:x + w])
            cv2.imshow('image', img)

        k = cv2.waitKey(100) & 0xFF == ord('s')
        if k == 10 or count >= 30:  # Save up to 30 images
            break

    cam.release()
    cv2.destroyAllWindows()
    return redirect(url_for('index'))

@app.route('/run-data-gather')
def run_data_gather():
    subprocess.Popen(["python", "datagathering.py"])  # Non-blocking call to gathering.py
    return "Data gather script started!"

# Modified route to accept both GET and POST methods
@app.route('/run-recognizer', methods=['GET', 'POST'])
def run_recognizer():
    subprocess.Popen(["python", "Recognizer.py"])  # Non-blocking call to recognizer.py
    return "Recognizer script started!"

@app.route('/criminal_list', methods=['GET'])
def criminal_list():
    # Load the images from the data folder
    images = os.listdir('data')
    images = [img for img in images if img.endswith(('.jpg', '.png'))]

    # Load criminal information from the CSV file
    criminal_info = {}
    with open('data/criminal_info.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            criminal_info[row[0]] = {'name': row[1], 'crime': row[2]}

    return render_template('criminal_list.html', images=images, criminal_info=criminal_info)

if __name__ == '__main__':
    app.run(debug=True)
