import streamlit as st
import pandas as pd
import torch
import openai
import os
import numpy as np
import json
from PIL import Image
from torchvision import models
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="Diabetic Foot Analysis System",
    page_icon="🏥",
    layout="wide",
    
)

# Konfigurasi OpenAI API Key
openai_api_key = st.secrets.get("OPENAI_API_KEY", None)

if openai_api_key:
    import openai
    openai.api_key = openai_api_key

# MobileNetV3 Model Definition
class MobileNetV3Model(nn.Module):
    def __init__(self, extractor_trainable: bool = True):
        super(MobileNetV3Model, self).__init__()
        mobilenet = models.mobilenet_v3_large(pretrained=True)
        self.feature_extractor = mobilenet.features
        self.fc = nn.Linear(mobilenet.classifier[0].in_features * 2, 1)

    def forward(self, left_image, right_image):
        x_left = self.feature_extractor(left_image)
        x_right = self.feature_extractor(right_image)
        x_left = F.adaptive_avg_pool2d(x_left, 1).reshape(x_left.size(0), -1)
        x_right = F.adaptive_avg_pool2d(x_right, 1).reshape(x_right.size(0), -1)
        x = torch.cat((x_left, x_right), dim=1)
        return self.fc(x)

# Fungsi untuk preprocess gambar
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.transpose(image, (2, 0, 1))
    return torch.tensor(image, dtype=torch.float32)


# Fungsi untuk halaman 1
def halaman_1():
    st.title("Deskripsi DFU dan Edukasi")
    st.image("DFU.png", caption="Contoh Diabetic Foot Ulcer (DFU)", use_column_width=True)
    st.header("Apa itu Diabetic Foot Ulcer (DFU)?")
    st.write("""
        Diabetic Foot Ulcer (DFU) adalah luka pada kaki yang disebabkan oleh diabetes yang dapat menyebabkan infeksi
        serius jika tidak ditangani dengan baik. Pengobatan dan pencegahan yang tepat dapat mengurangi risiko
        komplikasi lebih lanjut.
    """)
    st.header("Edukasi tentang DFU")
    st.write("""
        Penting bagi penderita diabetes untuk menjaga kesehatan kaki dengan cara rutin memeriksa kondisi kaki,
        menjaga kebersihan, dan mengenakan sepatu yang nyaman.
    """)

# Fungsi untuk halaman 2
def halaman_2():
    st.title("🏥 Advanced Diabetic Foot Analysis System")
    st.markdown("""
    This system combines advanced AI models to analyze foot images and provide comprehensive diabetic foot assessments.
    Upload clear images of both feet for the most accurate analysis.
    """)

    @st.cache_resource
    def load_model():
        model = MobileNetV3Model()
        model.load_state_dict(torch.load('mobilenet_v3_model.pth'), strict=False)
        model.eval()
        return model

    try:
        model = load_model()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Left Foot Image")
        uploaded_left_image = st.file_uploader("Upload Left Foot Image", type=["jpg", "png", "jpeg"])
        if uploaded_left_image:
            left_image = Image.open(uploaded_left_image)
            st.image(left_image, use_column_width=True)

    with col2:
        st.subheader("Right Foot Image")
        uploaded_right_image = st.file_uploader("Upload Right Foot Image", type=["jpg", "png", "jpeg"])
        if uploaded_right_image:
            right_image = Image.open(uploaded_right_image)
            st.image(right_image, use_column_width=True)

    if uploaded_left_image and uploaded_right_image:
        if st.button("Analyze Images", key="analyze_button"):
            with st.spinner("Analyzing images... Please wait."):
                try:
                    left_tensor = preprocess_image(left_image).unsqueeze(0)
                    right_tensor = preprocess_image(right_image).unsqueeze(0)
                    
                    with torch.no_grad():
                        prediction = model(left_tensor, right_tensor)
                        is_diabetic = prediction.item() > 0.5

                    prediction_label = "Diabetic" if is_diabetic else "Non-Diabetic"
                    prediction_probability = torch.sigmoid(prediction).item() * 100

                    st.markdown("### 📊 Classification Results")
                    results_col1, results_col2 = st.columns(2)
                    with results_col1:
                        st.metric("Prediction", prediction_label)
                    with results_col2:
                        st.metric("Confidence", f"{prediction_probability:.1f}%")

                    # Save analysis results to an Excel file
                    data = {
                        "Date": [str(pd.Timestamp.now())],
                        "Prediction": [prediction_label],
                        "Confidence": [f"{prediction_probability:.1f}%"]
                    }
                    df = pd.DataFrame(data)
                    
                    # Load or create analysis log
                    log_file = "analysis_log.xlsx"
                    if os.path.exists(log_file):
                        df_existing = pd.read_excel(log_file)
                        df = pd.concat([df_existing, df], ignore_index=True)
                    
                    df.to_excel(log_file, index=False)

                except Exception as e:
                    st.error(f"An error occurred during analysis: {str(e)}")

# Fungsi untuk halaman 3
def halaman_3():
    st.title("Data Pengunjung")
    st.write("""
        Halaman ini menampilkan data pengunjung yang telah mengakses aplikasi ini beserta hasil analisis dari gambar kaki.
    """)

    log_file = "analysis_log.xlsx"
    if os.path.exists(log_file):
        df = pd.read_excel(log_file)
        st.table(df)

        # Display Pie Chart for prediction results
        st.write("### Distribusi Hasil Prediksi")
        prediction_counts = df['Prediction'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(prediction_counts, labels=prediction_counts.index, autopct='%1.1f%%', startangle=90)
        ax.axis("equal")
        st.pyplot(fig)
    else:
        st.write("Belum ada data pengunjung yang tersedia.")

# Navigasi halaman
st.sidebar.title("Navigasi")
pilihan = st.sidebar.radio("Pilih Halaman:", ("Deskripsi DFU & Edukasi", "Real-Time Analysis", "Data Pengunjung"))

if pilihan == "Deskripsi DFU & Edukasi":
    halaman_1()
elif pilihan == "Real-Time Analysis":
    halaman_2()
elif pilihan == "Data Pengunjung":
    halaman_3()
