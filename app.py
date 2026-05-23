import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image

# page config
st.set_page_config(page_title="Face Morph Detection", layout="centered")

st.title("Face Morphing Attack Detection System")

st.write("Upload an image to check whether it is REAL or MORPHED")

# upload
uploaded_file = st.file_uploader("Choose Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    st.write("Processing...")

    # model
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    img = transform(img).unsqueeze(0)

    output = model(img)
    _, pred = torch.max(output, 1)

    if pred.item() == 0:
        st.success("✅ REAL FACE")
    else:
        st.error("❌ MORPHED FACE")