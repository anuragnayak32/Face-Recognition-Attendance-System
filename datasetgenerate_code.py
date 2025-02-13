import cv2
import os

# Function to crop faces from the frame
def face_cropped(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        return frame[y:y + h, x:x + w]
    return None

# Function to generate the dataset
def generate_dataset(student_name, student_id, base_folder="Dataset"):
    # Create the student-specific folder inside the base folder
    output_folder = os.path.join(base_folder, student_name)
    os.makedirs(output_folder, exist_ok=True)

    # Access the webcam
    cap = cv2.VideoCapture(0)
    img_id = 0

    while True:
        ret, myFrame = cap.read()
        if not ret:
            print("Failed to capture image. Please check your camera.")
            break

        cropped_face = face_cropped(myFrame)
        if cropped_face is not None:
            img_id += 1
            face = cv2.resize(cropped_face, (450, 450))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            # Save the image to the specified path
            file_name_path = os.path.join(output_folder, f"{student_name}.{student_id}.{img_id}.jpg")
            cv2.imwrite(file_name_path, face)

            # Display the captured image with the image ID
            cv2.putText(face, f"ID: {img_id}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow("Cropped Face", face)

        # Stop capturing after 100 images or if 'Enter' key is pressed
        if cv2.waitKey(1) == 13 or img_id == 200:  # 13 is the ASCII for Enter
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {img_id} images for {student_name} with ID: {student_id}")

# Call the function with a sample student name and ID
generate_dataset(student_name="Varshitha", student_id=1)
    