import cv2
import os
import numpy as np
from PIL import Image
from datetime import datetime

class AttendanceSystem:
    def __init__(self):
        self.recognized_students = {}  # Tracks recognized students within a session

    def mark_attendance(self, student_id, name, department):
        try:
            with open('attendance.csv', "a+", newline="\n") as f:
                f.seek(0)
                data = f.readlines()
                date_today = datetime.now().strftime("%d/%m/%Y")

                # Ensure student is marked only once per day
                if any(f"{student_id},{name},{department},{date_today}" in line for line in data):
                    return

                current_time = datetime.now().strftime("%H:%M:%S")
                f.write(f"{student_id},{name},{department},{current_time},{date_today},Present\n")
                
        except Exception as e:
            print(f"Error marking attendance: {e}")

    def get_student_info(self, student_id):
        # Replace this with a dynamic database or API integration for scalability
        student_db = {
            1: (400, "AdiSeshu", "CSE"),
            2: (498, "Anurag", "ECE"),
            3: (482, "Bibek", "CSE"),
            4: (485, "Deepak", "CSE"),
            5: (583, "Lakhan", "CSE"),
            6: (1045, "Lakshmi", "CSE"),
            7: (490, "Losta", "CSE"),
            8: (513, "Shrijal", "CSE"),
            9: (1018, "Varshitha", "CSE"),
            10: (1017, "Vyshnavi", "CSE")
        }
        return student_db.get(student_id, None)

    def face_recognize(self):
        def draw_boundary(img, classifier, scaleFactor, minNeighbors, clf, recognized_faces):
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)

            for (x, y, w, h) in features:
                face_roi = gray_img[y:y + h, x:x + w]
                face_roi = cv2.resize(face_roi, (100, 100))  # Resize to match training size
                face_roi = cv2.equalizeHist(face_roi)  # Histogram equalization for better contrast

                id_, confidence = clf.predict(face_roi)
                confidence_percentage = int(100 * (1 - confidence / 300))

                # Ensure confidence is above threshold
                if confidence_percentage > 65:  # Threshold for recognition
                    student_info = self.get_student_info(id_)
                    if student_info:
                        student_id, name, department = student_info

                        # Check if the student is already marked for this session
                        if student_id not in recognized_faces:
                            recognized_faces[student_id] = {
                                'name': name,
                                'department': department,
                                'confidence': confidence_percentage,
                                'marked': False
                            }

                        if not recognized_faces[student_id]['marked']:
                            # Mark attendance
                            self.mark_attendance(student_id, name, department)
                            recognized_faces[student_id]['marked'] = True

                        # Display recognized face
                        cv2.putText(img, f"ID: {student_id}", (x, y - 60), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 0), 2)
                        cv2.putText(img, f"Name: {name}", (x, y - 35), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 0), 2)
                        cv2.putText(img, f"Dept: {department}", (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 0), 2)
                else:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(img, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)

        def recognize(img, clf, face_cascade, recognized_faces):
            draw_boundary(img, face_cascade, 1.2, 5, clf, recognized_faces)  # Adjusted parameters

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        clf = cv2.face.LBPHFaceRecognizer_create()
        clf.read("classifier.xml")

        recognized_faces = {}  # Dictionary to store recognized faces and their attendance status for this session

        video_capture = cv2.VideoCapture(0)
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Failed to capture video. Exiting.")
                break
            recognize(frame, clf, face_cascade, recognized_faces)
            cv2.imshow("Face Recognition", frame)

            if cv2.waitKey(1) == 13:  # Exit on 'Enter'
                break

        video_capture.release()
        cv2.destroyAllWindows()



    def train_classifier(self):
        data_dir = "Dataset"  # Main dataset directory
        faces = []
        ids = []
        name_to_id = {"AdiSeshu": 1, "Anurag": 2, "Bibek": 3, "Deepak": 4, "Lakhan": 5, "Lakshmi": 6, "Losta": 7, "Shrijal": 8, "Varshitha": 9, "Vyshnavi": 10}  # Mapping folder names to IDs

        for folder in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, folder)
            if os.path.isdir(folder_path):  # Ensure it's a folder
                id_ = name_to_id.get(folder)
                if id_ is None:
                    print(f"Skipping unknown folder: {folder}")
                    continue

                for file in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file)
                    try:
                        img = Image.open(file_path).convert('L')  # Convert to grayscale
                        imageNp = np.array(img, 'uint8')
                        imageNp = cv2.equalizeHist(imageNp)  # Equalize histogram
                        faces.append(imageNp)
                        ids.append(id_)
                        cv2.imshow("Training", imageNp)
                        cv2.waitKey(10)
                    except Exception as e:
                        print(f"Skipped file {file_path}: {e}")

        if len(faces) < 2:
            print("Not enough data to train the classifier. Please add more samples.")
            return

        ids = np.array(ids)
        clf = cv2.face.LBPHFaceRecognizer_create()
        clf.train(faces, ids)
        clf.write("classifier.xml")
        cv2.destroyAllWindows()
        print("Training complete! Classifier saved as 'classifier.xml'.")


if __name__ == "__main__":
    attendance_system = AttendanceSystem()
    print("1. Train Classifier")
    print("2. Recognize Faces")

    choice = int(input("Enter your choice: "))
    if choice == 1:
        attendance_system.train_classifier()
    elif choice == 2:
        attendance_system.face_recognize()
    else:
        print("Invalid choice. Exiting.")
