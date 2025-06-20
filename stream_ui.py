import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision import transforms
from model import DigitCNN
import os

st.title("Handwritten Digit Recognition (MNIST, PyTorch)")

st.write("Draw a digit (0-9) below and click **Predict**.")

# Streamlit's canvas component
try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:
    st.error("Please install streamlit-drawable-canvas: pip install streamlit-drawable-canvas")
    st.stop()

canvas_result = st_canvas(
    fill_color="white",
    stroke_width=15,
    stroke_color="black",
    background_color="white",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DigitCNN().to(device)
    # Get absolute path to model file
    model_path = os.path.join(os.path.dirname(__file__), "digit_cnn.pth")
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please train the model and place 'digit_cnn.pth' here.")
        st.stop()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

if st.button("Predict"):
    if canvas_result.image_data is not None:
        # Convert RGBA to L (grayscale)
        img = Image.fromarray((255 - canvas_result.image_data[:, :, 0]).astype(np.uint8))
        img = img.resize((28, 28))
        img = ImageOps.invert(img)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        img_tensor = transform(img).unsqueeze(0)
        model, device = load_model()
        img_tensor = img_tensor.to(device)
        with torch.no_grad():
            output = model(img_tensor)
            pred = output.argmax(dim=1, keepdim=True).item()
        st.success(f"Prediction: {pred}")
    else:
        st.warning("Please draw a digit before predicting.")

st.write("Tip: Use your mouse or touchscreen to draw.")



