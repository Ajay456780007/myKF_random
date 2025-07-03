import customtkinter as ctk
import tkinter.filedialog as fd
import tkinter.messagebox as msg
import cv2
import numpy as np
import os
from threading import Thread
from PIL import Image
import pathlib
from datetime import datetime
import tensorflow as tf
import sys
import dlib
from scipy.spatial import distance as dist
import imutils

# Paths and model loading
if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(base_path, "proposed_model_3.h5")
predictor_path = os.path.join(base_path, "shape_predictor_68_face_landmarks.dat")

if not os.path.exists(model_path):
    msg.showerror("Model Error", f"Model file not found at: {model_path}")
    sys.exit(1)

if not os.path.exists(predictor_path):
    msg.showerror("Model Error", f"Facial landmark model not found at: {predictor_path}")
    sys.exit(1)

model = tf.keras.models.load_model(model_path)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor(predictor_path)
detector = dlib.get_frontal_face_detector()

# EAR functions
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

(L_START, L_END) = (42, 48)
(R_START, R_END) = (36, 42)
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 15

class DrowsinessApp:
    def __init__(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.app = ctk.CTk()
        self.app.title("Drowsiness Detection System")
        self.app.geometry("800x600")

        self.tab_view = ctk.CTkTabview(self.app, width=780, height=580)
        self.tab_view.pack(padx=10, pady=10)

        self.tab_home = self.tab_view.add("Home")
        self.tab_live = self.tab_view.add("Live Detection")
        self.tab_test = self.tab_view.add("Image Test")
        self.tab_about = self.tab_view.add("About")

        self.setup_home_tab()
        self.setup_live_tab()
        self.setup_image_test_tab()
        self.setup_about_tab()

    def setup_home_tab(self):
        ctk.CTkLabel(self.tab_home, text="Welcome to Drowsiness Detection System", font=("Arial", 22)).pack(pady=40)
        ctk.CTkLabel(self.tab_home, text="Navigate through tabs to use features", font=("Arial", 16)).pack(pady=10)

    def setup_live_tab(self):
        self.start_button = ctk.CTkButton(self.tab_live, text="Start Detection", command=self.start_detection_thread)
        self.start_button.pack(pady=20)
        self.stop_info_label = ctk.CTkLabel(self.tab_live, text="Press 'Q' to stop detection", font=("Arial", 14), text_color="gray")
        self.stop_info_label.pack(pady=5)

    def setup_image_test_tab(self):
        self.test_label = ctk.CTkLabel(self.tab_test, text="Upload an Image to Test Drowsiness", font=("Arial", 18))
        self.test_label.pack(pady=10)

        self.upload_button = ctk.CTkButton(self.tab_test, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=10)

        self.image_label = ctk.CTkLabel(self.tab_test, text="")
        self.image_label.pack(pady=10)

        self.status_label = ctk.CTkLabel(self.tab_test, text="", font=("Arial", 16))
        self.status_label.pack(pady=10)

        self.download_button = ctk.CTkButton(self.tab_test, text="Download Image", command=self.download_image)
        self.download_button.pack(pady=10)
        self.download_button.pack_forget()

    def setup_about_tab(self):
        about_text = (
            "This application detects driver drowsiness using:\n"
            "- Deep Learning Model\n"
            "- Eye Aspect Ratio (EAR) Detection\n\n"
            "Developed by: GUJAR"
        )
        ctk.CTkLabel(self.tab_about, text=about_text, font=("Arial", 14), justify="left").pack(pady=30)

    def start_detection_thread(self):
        thread = Thread(target=self.detect_drowsiness)
        thread.start()

    def predict_image(self, img):
        img = cv2.resize(img, (227, 227))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)
        pred = model.predict(img)[0]
        return np.argmax(pred), pred

    def detect_drowsiness(self):
        cap = cv2.VideoCapture(0)
        counter = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)

            label = "Analyzing..."
            color = (255, 255, 0)

            for rect in rects:
                shape = predictor(gray, rect)
                shape = np.array([[p.x, p.y] for p in shape.parts()])

                leftEye = shape[L_START:L_END]
                rightEye = shape[R_START:R_END]

                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0

                if ear < EAR_THRESHOLD:
                    counter += 1
                    if counter >= EAR_CONSEC_FRAMES:
                        label = "Drowsy"
                        color = (0, 0, 255)
                else:
                    counter = 0
                    label = "Not Drowsy"
                    color = (0, 255, 0)

                for (x, y) in np.concatenate((leftEye, rightEye), axis=0):
                    cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)

                break  # Only process first detected face

            cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imshow("Live Drowsiness Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    def upload_image(self):
        file_path = fd.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not file_path:
            self.status_label.configure(text="No image selected.", text_color="orange")
            return

        frame = cv2.imread(file_path)
        if frame is None:
            self.status_label.configure(text="Failed to load image.", text_color="red")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        label = "Analyzing..."
        color = (255, 255, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = np.array([[p.x, p.y] for p in shape.parts()])
            leftEye = shape[L_START:L_END]
            rightEye = shape[R_START:R_END]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            label = "Drowsy" if ear < EAR_THRESHOLD else "Not Drowsy"
            color = (0, 0, 255) if label == "Drowsy" else (0, 255, 0)
            break

        self.status_label.configure(text=f"Prediction: {label}", text_color="red" if label == "Drowsy" else "green")
        self.processed_frame = frame
        self.download_button.pack()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        ctk_img = ctk.CTkImage(light_image=img_pil, size=(600, 400))
        self.image_label.configure(image=ctk_img, text="")
        self.image_label.image = ctk_img

    def download_image(self):
        downloads_path = str(pathlib.Path.home() / "Downloads")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"drowsiness_result_{timestamp}.jpg"
        save_path = os.path.join(downloads_path, filename)

        try:
            cv2.imwrite(save_path, self.processed_frame)
            msg.showinfo("Success", f"Image saved to: {save_path}")
        except Exception as e:
            msg.showerror("Error", f"Failed to save image: {str(e)}")

    def run(self):
        self.app.mainloop()


if __name__ == "__main__":
    DrowsinessApp().run()
