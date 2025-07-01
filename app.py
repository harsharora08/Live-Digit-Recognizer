import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Digit Recognizer", layout="centered")

st.title("🖌️ Handwritten Digit Recognizer")

model = load_model("mnist.h5")

canvas_result = st_canvas(
    fill_color="white", 
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Predict"):
    if canvas_result.image_data is not None:
        img = canvas_result.image_data

        # Convert RGBA -> grayscale
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
        
        # Resize to 28x28
        img = cv2.resize(img, (28, 28))
        
        # Invert (white digit on black bg -> black digit on white bg)
        img = cv2.bitwise_not(img)
        
        # Normalize and reshape
        img = img.reshape(1, 28, 28, 1).astype(np.float32)

        pred = model.predict(img, verbose=0).argmax()
        st.success(f"✅ Predicted Digit: **{pred}**")
