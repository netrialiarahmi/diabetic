import streamlit as st
import pandas as pd
import torch
import openai
import os
import numpy as np
import json
from PIL import Image
from torchvision import models
from torchvision.models import MobileNet_V3_Large_Weights
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
import base64
from io import BytesIO
import tempfile
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Diabetic Foot Analysis System",
    page_icon="🏥",
    layout="centered",
)

# Custom CSS for improved UI styling, full-width navbar, and responsive layout
st.markdown(
    """
    <style>
    /* Full-width navbar container */
    .css-18ni7ap.e8zbici0 {
        display: flex;
        justify-content: center;
        width: 100%;
    }
    .navbar-container {
        width: 100%;
        max-width: 100%;
        display: flex;
        justify-content: center;
        padding: 5px 0;
        background-color: #0d6efd;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.25);
        border-radius: 8px;
    }
    /* Center align the main content with max width */
    .main-container {
        max-width: 1000px;
        margin: auto;
        padding: 20px;
    }
    /* Main title styling */
    .main-title {
        font-size: 2em;
        font-weight: bold;
        color: #333333;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 30px;
    }
    /* Section title styling */
    .section-title {
        font-size: 1.5em;
        font-weight: bold;
        color: #333333;
        margin-top: 30px;
        margin-bottom: 10px;
    }
    /* Justify the section content for better alignment */
    .section-content {
        font-size: 1.1em;
        line-height: 1.6;
        color: #555555;
        text-align: justify;
    }
    /* Responsive image container styling */
    .responsive-img-container {
        width: 100%;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .responsive-img-container img {
        width: 100%;
        height: auto;
        border-radius: 10px;
    }
    /* Caption text styling */
    .caption {
        font-style: italic;
        color: grey;
        margin-top: 5px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to display responsive images using HTML and CSS with full-width
def display_responsive_image(uploaded_file, caption):
    if isinstance(uploaded_file, str):  # If it's a file path
        with open(uploaded_file, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode()
    else:  # If it's an UploadedFile object
        img_base64 = base64.b64encode(uploaded_file.read()).decode()
        uploaded_file.seek(0)  # Reset file pointer for later use

    st.markdown(
        f"""
        <div class="responsive-img-container">
            <img src="data:image/png;base64,{img_base64}" alt="{caption}">
            <p class="caption">{caption}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# Function to preprocess images for model input
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.transpose(image, (2, 0, 1))
    return torch.tensor(image, dtype=torch.float32)




# MobileNetV3 Model Definition
class MobileNetV3Model(nn.Module):
    def __init__(self, extractor_trainable: bool = True):
        super(MobileNetV3Model, self).__init__()
        mobilenet = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        self.feature_extractor = mobilenet.features
        self.fc = nn.Linear(mobilenet.classifier[0].in_features * 2, 1)

    def forward(self, left_image, right_image):
        x_left = self.feature_extractor(left_image)
        x_right = self.feature_extractor(right_image)
        x_left = F.adaptive_avg_pool2d(x_left, 1).reshape(x_left.size(0), -1)
        x_right = F.adaptive_avg_pool2d(x_right, 1).reshape(x_right.size(0), -1)
        x = torch.cat((x_left, x_right), dim=1)
        return self.fc(x)

# Define functions for each page
def halaman_1():
    # Hero Section with Custom Styling
    st.markdown("""
        <div style="padding: 2rem; background: linear-gradient(135deg, #0d6efd20 0%, #0d6efd05 100%); border-radius: 1rem; margin-bottom: 2rem;">
            <h1 style="color: #0d6efd; font-size: 2.5rem; font-weight: bold; text-align: center; margin-bottom: 1rem;">
                Advanced Diabetic Foot Analysis System
            </h1>
            <p style="font-size: 1.2rem; color: #333; text-align: center; max-width: 800px; margin: 0 auto;">
                Selamat datang di platform yang dirancang khusus untuk membantu Anda menjaga kesehatan kaki dengan cara yang modern dan efektif.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Key Features Section using Columns
    st.markdown("<h2 style='text-align: center; color: #0d6efd; margin-bottom: 2rem;'>Fitur Utama</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div style="padding: 1.5rem; background-color: #fff; border-radius: 1rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1); height: 100%;">
                <h3 style="color: #0d6efd; font-size: 1.3rem; margin-bottom: 1rem;">🔍 Analisis Real-time</h3>
                <p style="color: #666;">Deteksi dini risiko komplikasi kaki diabetik menggunakan teknologi AI terkini.</p>
            </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
            <div style="padding: 1.5rem; background-color: #fff; border-radius: 1rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1); height: 100%;">
                <h3 style="color: #0d6efd; font-size: 1.3rem; margin-bottom: 1rem;">📊 Visualisasi Data</h3>
                <p style="color: #666;">Lihat dan pahami kondisi kaki Anda melalui visualisasi data yang mudah dimengerti.</p>
            </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
            <div style="padding: 1.5rem; background-color: #fff; border-radius: 1rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1); height: 100%;">
                <h3 style="color: #0d6efd; font-size: 1.3rem; margin-bottom: 1rem;">💡 Rekomendasi Personal</h3>
                <p style="color: #666;">Dapatkan saran perawatan yang dipersonalisasi berdasarkan hasil analisis.</p>
            </div>
        """, unsafe_allow_html=True)

    # Interactive Information Sections
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Example Image with Caption
    st.image("DFU.png", caption="Contoh Perkembangan Diabetic Foot Ulcer (DFU)", use_column_width=True)

    # Expandable Sections for Detailed Information
    with st.expander("🔍 Mengapa Analisis Kaki Penting?"):
        st.markdown("""
            <div style="padding: 1rem; background-color: #f8f9fa; border-radius: 0.5rem;">
                <p style="color: #333; line-height: 1.6;">
                Kaki adalah indikator kesehatan yang vital bagi penderita diabetes. Perubahan kecil pada kulit, 
                aliran darah, atau struktur kaki dapat menjadi tanda awal komplikasi serius seperti:
                </p>
                <ul style="color: #333; line-height: 1.6;">
                    <li>Ulkus diabetik</li>
                    <li>Gangguan sirkulasi</li>
                    <li>Neuropati diabetik</li>
                    <li>Infeksi</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    # Assessment Categories with Modern Cards
    st.markdown("<h2 style='text-align: center; color: #0d6efd; margin: 2rem 0;'>Kategori Penilaian</h2>", unsafe_allow_html=True)
    
    assessment_categories = {
        "Kesehatan Kulit": {
            "icon": "🔍",
            "description": "Analisis warna, tekstur, dan kondisi kulit kaki secara menyeluruh."
        },
        "Indikator Sirkulasi": {
            "icon": "🌡️",
            "description": "Pemeriksaan aliran darah dan tanda-tanda gangguan sirkulasi."
        },
        "Analisis Deformitas": {
            "icon": "👣",
            "description": "Evaluasi struktur kaki dan identifikasi area tekanan berlebih."
        },
        "Inspeksi Luka/Ulkus": {
            "icon": "🔬",
            "description": "Pemeriksaan keberadaan luka, tanda-tanda penyembuhan atau perburukan, kondisi jaringan sekitar, dan indikator infeksi."
        },
        "Kondisi Kuku": {
            "icon": "💅",
            "description": "Evaluasi warna, tekstur, pola pertumbuhan, tanda infeksi, dan abnormalitas ketebalan kuku."
        }
    }
    
    # Create rows of 3 columns each
    for i in range(0, len(assessment_categories), 3):
        cols = st.columns(3)
        # Get subset of categories for current row
        current_categories = dict(list(assessment_categories.items())[i:i+3])
        
        for idx, (category, info) in enumerate(current_categories.items()):
            with cols[idx]:
                st.markdown(f"""
                    <div style="padding: 1.5rem; background-color: #fff; border-radius: 1rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; height: 100%;">
                        <div style="font-size: 2rem; margin-bottom: 0.5rem;">{info['icon']}</div>
                        <h3 style="color: #0d6efd; font-size: 1.2rem; margin-bottom: 1rem;">{category}</h3>
                        <p style="color: #666; font-size: 0.9rem;">{info['description']}</p>
                    </div>
                """, unsafe_allow_html=True)

    # Personalized Recommendations Section
    st.markdown("<h2 style='text-align: center; color: #0d6efd; margin: 2rem 0;'>Rekomendasi Personal</h2>", unsafe_allow_html=True)
    
    recommendations = {
        "Tindakan Segera": {
            "icon": "🚨",
            "description": "Identifikasi kebutuhan perawatan mendesak dan tindakan medis yang diperlukan",
            "details": [
                "Evaluasi tingkat kegawatan berdasarkan kondisi luka",
                "Penanganan khusus sesuai tingkat keparahan",
                "Kebutuhan konsultasi dengan profesional medis"
            ],
            "dfu_context": "Deteksi dini DFU dapat mencegah amputasi hingga 85% kasus. Tindakan segera saat tanda awal muncul sangat kritis."
        },
        "Protokol Perawatan Harian": {
            "icon": "🧼",
            "description": "Panduan perawatan kaki harian yang disesuaikan dengan kondisi Anda",
            "details": [
                "Prosedur pembersihan yang aman dan efektif",
                "Rekomendasi pelembab khusus diabetik",
                "Rutinitas pemeriksaan mandiri",
                "Metode pengurangan tekanan pada area berisiko"
            ],
            "dfu_context": "Perawatan kaki yang tepat dapat menurunkan risiko DFU sebesar 50%. Rutinitas harian yang konsisten adalah kunci pencegahan."
        },
        "Strategi Pencegahan Risiko": {
            "icon": "🛡️",
            "description": "Langkah-langkah pencegahan yang disesuaikan dengan faktor risiko personal",
            "details": [
                "Rekomendasi alas kaki khusus diabetik",
                "Penyesuaian aktivitas fisik",
                "Pertimbangan lingkungan",
                "Tindakan pencegahan spesifik"
            ],
            "dfu_context": "77% kasus DFU dapat dicegah dengan strategi pencegahan yang tepat dan disesuaikan dengan kondisi pasien."
        },
        "Protokol Pemantauan": {
            "icon": "📊",
            "description": "Sistem pemantauan berkala untuk mencegah komplikasi",
            "details": [
                "Checklist pemeriksaan harian",
                "Tanda-tanda bahaya yang perlu diwaspadai",
                "Indikator untuk mencari bantuan medis",
                "Jadwal kontrol rutin"
            ],
            "dfu_context": "Pemantauan rutin dapat mendeteksi 89% tanda awal DFU sebelum berkembang menjadi kondisi serius."
        },
        "Penyesuaian Gaya Hidup": {
            "icon": "🌟",
            "description": "Rekomendasi perubahan gaya hidup untuk mendukung kesehatan kaki",
            "details": [
                "Program olahraga yang aman",
                "Pertimbangan diet khusus",
                "Modifikasi aktivitas sehari-hari",
                "Langkah-langkah perlindungan"
            ],
            "dfu_context": "Penyesuaian gaya hidup yang tepat dapat menurunkan risiko DFU hingga 60% dan mempercepat proses penyembuhan."
        }
    }
    
    for category, info in recommendations.items():
        with st.expander(f"{info['icon']} {category}", expanded=False):
            st.markdown(f"""
                <div style="padding: 1.5rem; background-color: #f8f9fa; border-radius: 1rem;">
                    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                        <span style="font-size: 1.5rem; margin-right: 0.5rem;">{info['icon']}</span>
                        <h3 style="color: #0d6efd; margin: 0;">{category}</h3>
                    </div>
                    <p style="color: #333; margin-bottom: 1rem;">{info['description']}</p>
                    
                    <div style="margin-top: 1.5rem;">
                        <h4 style="color: #0d6efd; margin-bottom: 1rem;">Komponen Utama:</h4>
                        <ul style="color: #333; margin-bottom: 1.5rem; list-style-type: disc; padding-left: 1.5rem;">
                            {''.join(f'<li style="margin-bottom: 0.5rem;">{detail}</li>' for detail in info['details'])}
                        </ul>
                    </div>
                    
                    <div style="background-color: #e7f0ff; padding: 1rem; border-radius: 0.5rem; margin-top: 1rem;">
                        <h4 style="color: #0d6efd; margin-bottom: 0.5rem;">Kaitan dengan DFU:</h4>
                        <p style="color: #333; margin: 0;">{info['dfu_context']}</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    # Visual Separator
    st.markdown("<hr style='margin: 2rem 0;'>", unsafe_allow_html=True)

    # Call-to-Action Section
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
        <div style="padding: 2rem; background: linear-gradient(135deg, #0d6efd15 0%, #0d6efd05 100%); border-radius: 1rem; text-align: center; margin-top: 2rem;">
            <h2 style="color: #0d6efd; margin-bottom: 1rem;">Mulai Analisis Sekarang</h2>
            <p style="color: #333; margin-bottom: 1.5rem;">
                Lakukan pemeriksaan kaki Anda dan dapatkan rekomendasi personal dalam hitungan menit.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Footer Information
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 2rem 0;">
            <p>© 2024 Advanced Diabetic Foot Analysis System</p>
            <p style="font-size: 0.9rem;">Powered by AI & Medical Expertise</p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)  # Close main-container div

# Function for page 2 (Real-Time Analysis)
def halaman_2():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    st.markdown('<div class="main-title">🏥 Advanced Diabetic Foot Analysis System</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-content">
        This system combines advanced AI models and OpenAI Vision to analyze foot images and provide comprehensive diabetic foot assessments.
        Upload clear images of both feet for the most accurate analysis.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Add image upload instructions
    st.markdown(
        """
        <div style="border: 1px solid #ddd; padding: 15px; border-radius: 8px; background-color: #f9f9f9; margin-bottom: 20px;">
            <strong>Instructions for Uploading Images:</strong>
            <ul style="line-height: 1.6;">
                <li>Ensure both images are well-lit, clear, and focused.</li>
                <li>Position your foot on a flat, neutral background (e.g., white or light-colored floor).</li>
                <li>Take separate images of your <strong>left</strong> and <strong>right</strong> foot from a similar angle.</li>
                <li>Refer to the example images below to see the correct orientation and positioning for analysis.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Display example images for guidance
    col1, col2 = st.columns(2)
    with col1:
        st.image("CG010_M_L-rotated1-rotated1.png", caption="Example: Left Foot Image", use_container_width=True)
    with col2:
        st.image("CG010_M_R-rotated1-rotated1.png", caption="Example: Right Foot Image", use_container_width=True)

    def analyze_image_with_openai(image, context=""):
        """Analyze image using OpenAI Vision API with improved prompting"""
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        client = openai.OpenAI()
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert podiatrist and diabetic foot specialist with extensive experience in diabetic foot pressure maps analysis.
                    IMPORTANT: Always provide direct analysis of the image shown.
                    DO NOT start with 'I'm unable to analyze' or similar disclaimers
                    Your analysis should be:
                    1. Highly detailed and specific
                    2. Based on visible evidence in the image
                    3. Focused on diabetic-relevant indicators
                    4. Professional yet clear
                    5. Structured and methodical"""
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""Please analyze this foot image in detail, focusing on:

1. Skin Health Assessment:
   - Color variations and patterns
   - Texture abnormalities
   - Dryness levels
   - Any breaks or damages
   - Presence of calluses

2. Circulation Indicators:
   - Color distribution patterns
   - Any signs of reduced blood flow
   - Presence of swelling
   - Temperature indicators (if visible)

3. Deformity Analysis:
   - Foot structure alignment
   - Pressure point locations
   - Joint positions and angles
   - Arch characteristics

4. Wound/Ulcer Inspection:
   - Presence of any wounds
   - Signs of healing or deterioration
   - Surrounding tissue condition
   - Infection indicators

5. Nail Condition:
   - Color and texture
   - Growth patterns
   - Signs of infection
   - Thickness abnormalities

Additional Context: {context}"""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_str}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error in image analysis: {str(e)}"

    def generate_recommendations(classification_result, left_analysis, right_analysis):
        """Generate comprehensive recommendations based on analyses"""
        client = openai.OpenAI()
        
        recommendations_prompt = f"""Based on the following detailed foot analyses, provide specific care recommendations:

Classification: {classification_result}

Left Foot Analysis:
{left_analysis}

Right Foot Analysis:
{right_analysis}

Please provide detailed recommendations in these categories:

1. Immediate Actions Required:
   - Urgent care needs
   - Specific treatments
   - Professional consultations needed

2. Daily Care Protocol:
   - Cleaning procedures
   - Moisturizing recommendations
   - Inspection routine
   - Pressure relief methods

3. Risk Prevention Strategy:
   - Footwear recommendations
   - Activity modifications
   - Environmental considerations
   - Preventive measures

4. Monitoring Protocol:
   - What to check daily
   - Warning signs to watch
   - When to seek immediate care
   - Follow-up schedule

5. Lifestyle Adjustments:
   - Exercise recommendations
   - Dietary considerations
   - Daily activity modifications
   - Protective measures"""

        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a senior podiatrist specializing in diabetic foot care. Provide comprehensive, evidence-based recommendations that are practical and actionable."
                    },
                    {"role": "user", "content": recommendations_prompt}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating recommendations: {str(e)}"

    @st.cache_resource
    def load_model():
        model = MobileNetV3Model()
        model.load_state_dict(torch.load('mobilenet_v3_model.pth', weights_only=True), strict=False)
        model.eval()
        return model

    model = load_model()

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Left Foot Image")
        uploaded_left_image = st.file_uploader("Upload Left Foot Image", type=["jpg", "png", "jpeg"])
        if uploaded_left_image:
            st.image(uploaded_left_image, caption="Left Foot Image", use_container_width=True)

    with col2:
        st.subheader("Upload Right Foot Image")
        uploaded_right_image = st.file_uploader("Upload Right Foot Image", type=["jpg", "png", "jpeg"])
        if uploaded_right_image:
            st.image(uploaded_right_image, caption="Right Foot Image", use_container_width=True)

    if uploaded_left_image and uploaded_right_image:
        if st.button("Analyze Images", key="analyze_button"):
            with st.spinner("Analyzing images... Please wait."):
                try:
                    # Reset file pointers to read images for processing
                    left_image = Image.open(uploaded_left_image)
                    right_image = Image.open(uploaded_right_image)

                    # Preprocess images for the model
                    left_tensor = preprocess_image(left_image).unsqueeze(0)
                    right_tensor = preprocess_image(right_image).unsqueeze(0)

                    with torch.no_grad():
                        prediction = model(left_tensor, right_tensor)
                        is_diabetic = prediction.item() > 0.5

                    prediction_label = "Diabetic" if is_diabetic else "Non-Diabetic"
                    prediction_probability = torch.sigmoid(prediction).item() * 100

                    st.markdown('<div class="section-title">📊 Classification Results</div>', unsafe_allow_html=True)
                    results_col1, results_col2 = st.columns(2)
                    with results_col1:
                        st.metric("Prediction", prediction_label)
                    with results_col2:
                        st.metric("Confidence", f"{prediction_probability:.1f}%")

                    # OpenAI Vision Analysis
                    st.markdown("### 🔍 Detailed Analysis")
                    analysis_col1, analysis_col2 = st.columns(2)
                    
                    with analysis_col1:
                        st.markdown("#### Left Foot Analysis")
                        left_analysis = analyze_image_with_openai(
                            left_image, 
                            f"Left foot image. Model prediction: {prediction_label}"
                        )
                        st.write(left_analysis)
                    
                    with analysis_col2:
                        st.markdown("#### Right Foot Analysis")
                        right_analysis = analyze_image_with_openai(
                            right_image,
                            f"Right foot image. Model prediction: {prediction_label}"
                        )
                        st.write(right_analysis)

                    # Generate and display recommendations
                    st.markdown("### 💡 Care Recommendations")
                    recommendations = generate_recommendations(
                        prediction_label,
                        left_analysis,
                        right_analysis
                    )
                    st.write(recommendations)

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
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close main-container div

def halaman_3():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    st.markdown('<div class="main-title">Ringkasan Data Analisis</div>', unsafe_allow_html=True)

    # Read the Excel file
    log_file = "analysis_log.xlsx"
    if not os.path.exists(log_file):
        st.warning("Belum ada data analisis yang tersedia.")
        return

    df = pd.read_excel(log_file)
    
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Process Confidence column to numeric
    df['Confidence_Value'] = df['Confidence'].str.rstrip('%').astype(float)

    # Statistik Utama dengan Dua Kolom
    st.markdown("### 📊 Statistik Utama")
    col1, col2 = st.columns(2)
    if 'Prediction' in df.columns:
        diabetic_percent = (df['Prediction'] == 'Diabetic').mean() * 100
    else:
        diabetic_percent = 0.0
    
    if 'Confidence_Value' in df.columns:
        avg_confidence = df['Confidence_Value'].mean()
        high_conf = (df['Confidence_Value'] > 95).sum()
    else:
        avg_confidence = 0.0
        high_conf = 0
    
    # Kolom Pertama: Total Pengunjung dan Persentase Diabetic
    with col1:
        st.markdown(
            f"""
            <div style="padding: 15px; margin-bottom: 20px; border-radius: 10px; background-color: #f0f2f6; text-align: center;">
                <h3 style="margin: 0; color: #1f77b4;">Total Pengunjung</h3>
                <p style="font-size: 24px; margin: 10px 0;">{len(df)}</p>
            </div>
            <div style="padding: 15px; margin-bottom: 20px; border-radius: 10px; background-color: #f0f2f6; text-align: center;">
                <h3 style="margin: 0; color: #ff7f0e;">Persentase Diabetic</h3>
                <p style="font-size: 24px; margin: 10px 0;">{diabetic_percent:.1f}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    diabetic_percent = (df['Prediction'] == 'Diabetic').mean() * 100
    # Kolom Kedua: Confidence Rata-rata dan Confidence >95%
    with col2:
        st.markdown(
            f"""
            <div style="padding: 15px; margin-bottom: 20px; border-radius: 10px; background-color: #f0f2f6; text-align: center;">
                <h3 style="margin: 0; color: #2ca02c;">Confidence Rata-rata</h3>
                <p style="font-size: 24px; margin: 10px 0;">{avg_confidence:.1f}%</p>
            </div>
            <div style="padding: 15px; margin-bottom: 20px; border-radius: 10px; background-color: #f0f2f6; text-align: center;">
                <h3 style="margin: 0; color: #d62728;">Confidence >95%</h3>
                <p style="font-size: 24px; margin: 10px 0;">{high_conf}</p>
            </div>
            """,
            unsafe_allow_html=True
        )


    # 2. Advanced Visualizations Section
    st.markdown("### 📈 Visualisasi Data")
    
    # Create two columns for charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("#### Distribusi Prediksi")
        fig_pie = {
            "data": [{
                "values": df['Prediction'].value_counts().values,
                "labels": df['Prediction'].value_counts().index,
                "type": "pie",
                "hole": 0.4,
                "marker": {"colors": ["#ff7f0e", "#1f77b4"]}
            }],
            "layout": {
                "showlegend": True,
                "legend": {"orientation": "h"},
                "annotations": [{
                    "text": "Distribusi<br>Total",
                    "showarrow": False,
                    "font": {"size": 14}
                }]
            }
        }
        st.plotly_chart(fig_pie, use_container_width=True)

    with chart_col2:
        st.markdown("#### Distribusi Confidence")
        # Create confidence bins
        bins = [0, 50, 75, 95, 100]
        labels = ['0-50%', '51-75%', '76-95%', '>95%']
        df['Confidence_Bin'] = pd.cut(df['Confidence_Value'], bins=bins, labels=labels, right=True)
        confidence_dist = df['Confidence_Bin'].value_counts().sort_index()
        
        fig_bar = {
            "data": [{
                "x": confidence_dist.index,
                "y": confidence_dist.values,
                "type": "bar",
                "marker": {"color": "#2ca02c"}
            }],
            "layout": {
                "xaxis": {"title": "Confidence Range"},
                "yaxis": {"title": "Number of Predictions"},
                "showlegend": False
            }
        }
        st.plotly_chart(fig_bar, use_container_width=True)

    # 3. Time Series Analysis
    st.markdown("### 📅 Analisis Waktu")
    
    # Prepare time series data
    df['Date_Only'] = df['Date'].dt.date
    time_series = df.groupby(['Date_Only', 'Prediction']).size().unstack(fill_value=0)
    
    fig_line = {
        "data": [
            {
                "x": time_series.index,
                "y": time_series['Diabetic'] if 'Diabetic' in time_series.columns else [],
                "name": "Diabetic",
                "type": "scatter",
                "mode": "lines+markers",
                "marker": {"color": "#ff7f0e"}
            },
            {
                "x": time_series.index,
                "y": time_series['Non-Diabetic'] if 'Non-Diabetic' in time_series.columns else [],
                "name": "Non-Diabetic",
                "type": "scatter",
                "mode": "lines+markers",
                "marker": {"color": "#1f77b4"}
            }
        ],
        "layout": {
            "xaxis": {"title": "Date"},
            "yaxis": {"title": "Number of Predictions"},
            "legend": {"orientation": "h"},
            "hovermode": "x unified"
        }
    }
    st.plotly_chart(fig_line, use_container_width=True)

    # 4. Interactive Data Table
# 4. Interactive Data Table
    st.markdown("### 📋 Data Detail")
    
    # Add date filter
    date_min = df['Date'].min().date()
    date_max = df['Date'].max().date()
    selected_date_range = st.date_input(
        "Pilih Rentang Tanggal",
        value=(date_min, date_max),
        min_value=date_min,
        max_value=date_max
    )
    
    # Function to determine risk level based on confidence value and prediction
    def determine_risk(row):
        confidence = row['Confidence_Value']
        prediction = row['Prediction']
        
        if prediction == 'Diabetic':
            if confidence >= 90:
                return 'High Risk'
            elif confidence >= 70:
                return 'Moderate Risk'
            else:
                return 'Low Risk'
        else:  # Non-Diabetic
            if confidence >= 90:
                return 'Low Risk'
            elif confidence >= 70:
                return 'Low-Moderate Risk'
            else:
                return 'Moderate Risk'
    
    if len(selected_date_range) == 2:
        start_date, end_date = selected_date_range
        mask = (df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)
        filtered_df = df[mask].copy()
        
        # Add risk level
        filtered_df['Risk'] = filtered_df.apply(determine_risk, axis=1)
        
        # Format date and select only required columns
        display_df = filtered_df[['Date', 'Prediction', 'Risk']].copy()
        display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Create a color mapping for risk levels
        risk_colors = {
            'High Risk': '#ff4444',
            'Moderate Risk': '#ffbb33',
            'Low-Moderate Risk': '#99cc00',
            'Low Risk': '#00C851'
        }
        
        # Apply styling to the dataframe
        styled_df = display_df.style.apply(lambda x: [
            f'background-color: {risk_colors[val]};' if col == 'Risk' else '' 
            for val in x
        ], axis=1)
        
        st.dataframe(
            display_df,
            column_config={
                "Date": "Tanggal",
                "Prediction": "Prediksi",
                "Risk": "Risk Level"
            },
            hide_index=True
        )
    st.markdown('</div>', unsafe_allow_html=True)  # Close main-container div
# Define the option menu for navigation inside a full-width container
selections = option_menu(
    menu_title=None,
    options=['Home', "Real-Time Analysis", "User Dashboard"],
    icons=['house-fill', "file-earmark-medical-fill", "people-fill"],
    menu_icon="cast",
    default_index=0,
    orientation='horizontal',
    styles={
        "container": {
            "padding": "5px 0",
            "background-color": "#0d6efd",
            "border-radius": "8px",
            "width": "100%",
            "display": "flex",
            "justify-content": "center",
        },
        "icon": {"color": "#f9fafb", "font-size": "18px"},
        "hr": {"color": "#0d6dfdbe"},
        "nav-link": {
            "color": "#f9fafb",
            "font-size": "15px",
            "text-align": "center",
            "margin": "0 10px",
            "--hover-color": "#0761e97e",
            "padding": "10px 10px",
            "border-radius": "16px",
        },
        "nav-link-selected": {
            "background-color": "#ffd700",
            "color": "#0d6efd",
            "font-weight": "bold",
            "border-radius": "8px",  # Slightly smaller radius for a tighter highlight
            "padding": "10px 15px",
            "box-shadow": "0px 4px 10px rgba(0, 0, 0, 0.15)",  # Subtle shadow for depth
            "transition": "background-color 0.3s ease",  # Smooth transition for better UX
        }
    }
)


# Page selection
if selections == "Home":
    halaman_1()
elif selections == "Real-Time Analysis":
    halaman_2()
elif selections == "User Dashboard":
    halaman_3()
