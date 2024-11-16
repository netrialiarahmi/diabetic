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
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    st.markdown('<div class="main-title">Deskripsi DFU dan Edukasi</div>', unsafe_allow_html=True)
    display_responsive_image("DFU.png", "Contoh Diabetic Foot Ulcer (DFU)")
    
    st.markdown('<div class="section-title">Apa itu Diabetic Foot Ulcer (DFU)?</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-content">
        Diabetic Foot Ulcer (DFU) adalah luka pada kaki yang disebabkan oleh diabetes yang dapat menyebabkan infeksi
        serius jika tidak ditangani dengan baik. Pengobatan dan pencegahan yang tepat dapat mengurangi risiko
        komplikasi lebih lanjut.
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown('<div class="section-title">Peta Tekanan Kaki untuk Pencegahan DFU</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-content">
        Peta tekanan kaki dapat membantu mendeteksi area dengan tekanan tinggi pada kaki, yang mungkin menunjukkan risiko lebih tinggi terhadap ulserasi pada individu dengan diabetes. 
        Pemantauan rutin dan langkah-langkah pencegahan dapat membantu mengurangi kemungkinan berkembangnya DFU.
        </div>
        """,
        unsafe_allow_html=True
    )

    
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
    
    if len(selected_date_range) == 2:
        start_date, end_date = selected_date_range
        mask = (df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)
        filtered_df = df[mask].copy()
        filtered_df['Date'] = filtered_df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        st.dataframe(
            filtered_df,
            column_config={
                "Date": "Tanggal",
                "Prediction": "Prediksi",
                "Confidence": "Confidence"
            },
            hide_index=True
        )
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close main-container div
# Define the option menu for navigation inside a full-width container
selections = option_menu(
    menu_title=None,
    options=['Deskripsi DFU & Edukasi', "Real-Time Analysis", "Data Pengunjung"],
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
if selections == "Deskripsi DFU & Edukasi":
    halaman_1()
elif selections == "Real-Time Analysis":
    halaman_2()
elif selections == "Data Pengunjung":
    halaman_3()
