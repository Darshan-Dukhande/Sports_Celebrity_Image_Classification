import streamlit as st
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Load your model
model = load_model('best_model.h5')
face_cascade = cv2.CascadeClassifier(r"C:\Users\asus\Desktop\DS!ML\Image Classification\opencv\haarcascades\haarcascade_frontalface_default.xml")
class_labels = sorted(os.listdir('./cropped/'))  # get class names from training folders

st.set_page_config(page_title="Celebrity Classifier", layout="centered")
st.title("🏏 Celebrity Face Classifier")
st.write("Upload a photo and the model will try to recognize the celebrity.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

def detect_and_crop_face(image):
    """Detect face from uploaded image and return ROI"""
    image_np = np.array(image.convert('RGB'))
    img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = img_bgr[y:y+h, x:x+w]
        return roi
    return None

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner('Analyzing image...'):
        roi = detect_and_crop_face(image)
        if roi is not None:
            roi_resized = cv2.resize(roi, (224, 224))
            roi_array = img_to_array(roi_resized) / 255.0
            roi_array = np.expand_dims(roi_array, axis=0)

            prediction = model.predict(roi_array)[0]
            top_idx = np.argmax(prediction)
            predicted_label = class_labels[top_idx]

            st.success(f"🎯 Predicted: **{predicted_label}**")
        else:
            st.error("😢 No face with 2 eyes detected in the image.")



