import cv2
import os
import csv

# Set up the face detector
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

# Create the data directory if it doesn't exist
if not os.path.exists("data"):
    os.makedirs("data")

# Ask for user details: name, crime, and user ID
user_id = input("\nEnter user ID: ")
user_name = input("Enter name: ")
user_crime = input("Enter crime: ")

# Initialize image count
count = 0

# Open CSV to store user details (optional, for later use)
csv_file = "criminal_info.csv"
header = ['id', 'name', 'crime']
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

# Capture 5 images and save them with the user ID
while True:
    _, img = cam.read()
    img = cv2.flip(img, 1)  # Flip the image vertically
    faces = detector.detectMultiScale(img, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imwrite(f"data/{user_id}_{count}.jpg", img[y:y + h, x:x + w])  # Save the face image
        cv2.imshow('image', img)

        if count >= 5:  # Stop after 5 images
            break

    if count >= 5:
        break

# Save user details in the CSV file
with open(csv_file, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([user_id, user_name, user_crime])

# Release the camera and close the window
cam.release()
cv2.destroyAllWindows()
