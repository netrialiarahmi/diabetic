import streamlit as st
import streamlit.components.v1 as components
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
    # Apply custom font styles
    st.markdown("""
        <style>
        body {
            font-family: 'Arial', sans-serif;
        }
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Roboto', sans-serif;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Hero Section with Custom Styling
    components.html("""
        <div style="padding: 2rem; background: linear-gradient(135deg, #0d6efd20 0%, #0d6efd05 100%); border-radius: 1rem; margin-bottom: 2rem;">
            <h1 style="color: #0d6efd; font-size: 2.5rem; font-weight: bold; text-align: center; margin-bottom: 1rem;">
                Advanced Diabetic Foot Analysis System
            </h1>
            <p style="font-size: 1.2rem; color: #333; text-align: center; max-width: 800px; margin: 0 auto;">
                Welcome to a platform specifically designed to help you maintain foot health in a modern and effective way.
            </p>
        </div>
    """, height=250)
    
    # Key Features Section using Columns
    components.html("<h2 style='text-align: center; color: #0d6efd; margin-bottom: 2rem;'>Key Features</h2>", height=60)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        components.html("""
            <div style="padding: 1.5rem; background-color: #fff; border-radius: 1rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1); height: 100%;">
                <h3 style="color: #0d6efd; font-size: 1.3rem; margin-bottom: 1rem;">🔍 Real-time Analysis</h3>
                <p style="color: #666;">Early detection of diabetic foot complications using the latest AI technology.</p>
            </div>
        """, height=200)
                
    with col2:
        components.html("""
            <div style="padding: 1.5rem; background-color: #fff; border-radius: 1rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1); height: 100%;">
                <h3 style="color: #0d6efd; font-size: 1.3rem; margin-bottom: 1rem;">📊 User Summary</h3>
                <p style="color: #666;">View and understand user summary for the next advanced research purpose.</p>
            </div>
        """, height=200)
                
    with col3:
        components.html("""
            <div style="padding: 1.5rem; background-color: #fff; border-radius: 1rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1); height: 150%;">
                <h3 style="color: #0d6efd; font-size: 1.3rem; margin-bottom: 1rem;">💡 Personalized Recommendations</h3>
                <p style="color: #666;">Get personalized care suggestions based on analysis results.</p>
            </div>
        """, height=200)
    
    # Add extra space between Key Features and the DFU image
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Example Image with Caption
    st.image("DFU.png", caption="Example of Diabetic Foot Ulcer (DFU) Progression", use_column_width=True)
    
    # Expandable Sections for Detailed Information
    with st.expander("🔍 Why is Foot Analysis Important?"):
        components.html("""
            <div style="padding: 1rem; background-color: #f8f9fa; border-radius: 0.5rem;">
                <p style="color: #333; line-height: 1.6;">
                Feet are vital health indicators for people with diabetes. Small changes in skin, blood flow, or foot structure can be early signs of serious complications such as:
                </p>
                <ul style="color: #333; line-height: 1.6;">
                    <li>Diabetic ulcers</li>
                    <li>Circulatory disorders</li>
                    <li>Diabetic neuropathy</li>
                    <li>Infection</li>
                </ul>
            </div>
        """, height=200)
    
    # Assessment Categories with Modern Cards
    components.html("<h2 style='text-align: center; color: #0d6efd; margin: 2rem 0;'>Assessment Categories</h2>", height=60)
    
    assessment_categories = {
        "Skin Health": {
            "icon": "🔍",
            "description": "Comprehensive analysis of foot skin color, texture, and condition."
        },
        "Circulation Indicators": {
            "icon": "🌡️",
            "description": "Examination of blood flow and signs of circulatory disorders."
        },
        "Deformity Analysis": {
            "icon": "👣",
            "description": "Evaluation of foot structure and identification of areas with excessive pressure."
        },
        "Wound/Ulcer Inspection": {
            "icon": "🔬",
            "description": "Examination for the presence of wounds, signs of healing or deterioration, condition of surrounding tissue, and infection indicators."
        },
        "Nail Condition": {
            "icon": "💅",
            "description": "Evaluation of nail color, texture, growth patterns, signs of infection, and abnormalities in nail thickness."
        }
    }
    
    # Create rows of 3 columns each
    categories = list(assessment_categories.items())
    for i in range(0, len(categories), 3):
        cols = st.columns(3)
        current_categories = categories[i:i+3]
        
        for idx, (category, info) in enumerate(current_categories):
            with cols[idx]:
                components.html(f"""
                    <div style="padding: 1.5rem; background-color: #fff; border-radius: 1rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; height: 100%;">
                        <div style="font-size: 2rem; margin-bottom: 0.5rem;">{info['icon']}</div>
                        <h3 style="color: #0d6efd; font-size: 1.2rem; margin-bottom: 1rem;">{category}</h3>
                        <p style="color: #666; font-size: 0.9rem;">{info['description']}</p>
                    </div>
                """, height=250)
    
    # Personalized Recommendations Section
    components.html("<h2 style='text-align: center; color: #0d6efd; margin: 2rem 0;'>Personalized Recommendations</h2>", height=60)
    
    recommendations = {
        "Immediate Actions": {
            "icon": "🚨",
            "description": "Identify urgent care needs and required medical actions",
            "details": [
                "Evaluate urgency level based on wound condition",
                "Special handling according to severity",
                "Need for consultation with medical professionals"
            ],
            "dfu_context": "Early detection of DFU can prevent amputation in up to 85% of cases. Immediate action when early signs appear is critical."
        },
        "Daily Care Protocol": {
            "icon": "🧼",
            "description": "Daily foot care guidelines tailored to your condition",
            "details": [
                "Safe and effective cleaning procedures",
                "Recommendations for diabetic-specific moisturizers",
                "Self-examination routines",
                "Methods to reduce pressure on risk areas"
            ],
            "dfu_context": "Proper foot care can reduce the risk of DFU by 50%. Consistent daily routines are key to prevention."
        },
        "Risk Prevention Strategies": {
            "icon": "🛡️",
            "description": "Preventive steps tailored to personal risk factors",
            "details": [
                "Recommendations for diabetic-specific footwear",
                "Adjustment of physical activities",
                "Environmental considerations",
                "Specific preventive measures"
            ],
            "dfu_context": "77% of DFU cases can be prevented with appropriate and personalized prevention strategies."
        },
        "Monitoring Protocol": {
            "icon": "📊",
            "description": "Regular monitoring system to prevent complications",
            "details": [
                "Daily examination checklist",
                "Warning signs to watch out for",
                "Indicators to seek medical help",
                "Routine control schedules"
            ],
            "dfu_context": "Routine monitoring can detect 89% of early DFU signs before developing into serious conditions."
        },
        "Lifestyle Adjustments": {
            "icon": "🌟",
            "description": "Recommendations for lifestyle changes to support foot health",
            "details": [
                "Safe exercise programs",
                "Special dietary considerations",
                "Modifications of daily activities",
                "Protective measures"
            ],
            "dfu_context": "Appropriate lifestyle adjustments can reduce DFU risk by up to 60% and accelerate the healing process."
        }
    }
    
    for category, info in recommendations.items():
        with st.expander(f"{info['icon']} {category}", expanded=False):
            components.html(f"""
                <div style="padding: 1.5rem; background-color: #f8f9fa; border-radius: 1rem;">
                    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                        <span style="font-size: 1.5rem; margin-right: 0.5rem;">{info['icon']}</span>
                        <h3 style="color: #0d6efd; margin: 0;">{category}</h3>
                    </div>
                    <p style="color: #333; margin-bottom: 1rem;">{info['description']}</p>
                    
                    <div style="margin-top: 1.5rem;">
                        <h4 style="color: #0d6efd; margin-bottom: 1rem;">Main Components:</h4>
                        <ul style="color: #333; margin-bottom: 1.5rem; list-style-type: disc; padding-left: 1.5rem;">
                            {''.join(f'<li style="margin-bottom: 0.5rem;">{detail}</li>' for detail in info['details'])}
                        </ul>
                    </div>
                    
                    <div style="background-color: #e7f0ff; padding: 1rem; border-radius: 0.5rem; margin-top: 1rem;">
                        <h4 style="color: #0d6efd; margin-bottom: 0.5rem;">Relation to DFU:</h4>
                        <p style="color: #333; margin: 0;">{info['dfu_context']}</p>
                    </div>
                </div>
            """, height=400)
    
    # Visual Separator
    components.html("<hr style='margin: 2rem 0;'>", height=20)
    
    # Call-to-Action Section
    components.html("""
        <div style="padding: 2rem; background: linear-gradient(135deg, #0d6efd15 0%, #0d6efd05 100%); border-radius: 1rem; text-align: center; margin-top: 2rem;">
            <h2 style="color: #0d6efd; margin-bottom: 1rem;">Start Analysis Now</h2>
            <p style="color: #333; margin-bottom: 1.5rem;">
                Check your feet and get personalized recommendations in minutes.
            </p>
        </div>
    """, height=250)
    
    # Footer Information
    components.html("""
        <div style="text-align: center; color: #666; padding: 2rem 0;">
            <p>© 2024 Advanced Diabetic Foot Analysis System</p>
            <p style="font-size: 0.9rem;">Powered by AI & Medical Expertise</p>
        </div>
    """, height=100)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close main-container div

# Image preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image)

def analyze_image_with_openai(image, context=""):
    """Analyze image using OpenAI Vision API with detailed medical prompting"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    client = openai.OpenAI()
    
    try:
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert podiatrist and diabetic foot specialist with extensive experience in diabetic foot assessment. 
                    
Key Analysis Requirements:
1. Provide direct, clear analysis without disclaimers
2. Focus on medically relevant observations
3. Use professional medical terminology while remaining understandable
4. Structure analysis systematically
5. Highlight urgent concerns if present"""
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""Conduct a comprehensive diabetic foot analysis focusing on these key areas:

1. Skin Health Assessment:
   - Coloration patterns and abnormalities
   - Skin texture and integrity
   - Presence of dryness or scaling
   - Evidence of breaks or fissures
   - Location and severity of calluses
   - Signs of pressure points

2. Circulation Evaluation:
   - Color distribution patterns
   - Evidence of peripheral circulation
   - Presence of edema or swelling
   - Temperature variation indicators
   - Capillary refill indicators

3. Structural Analysis:
   - Foot alignment and architecture
   - Pressure point distribution
   - Joint positioning and angles
   - Arch characteristics and abnormalities
   - Toe positioning and spacing

4. Wound/Ulcer Detection:
   - Presence of active wounds
   - Signs of previous ulceration
   - Tissue condition around problem areas
   - Early warning signs of ulcer formation
   - Infection indicators

5. Nail Health Analysis:
   - Nail color and texture
   - Growth patterns and deformities
   - Signs of fungal infection
   - Thickness abnormalities
   - Surrounding tissue condition

Additional Clinical Context: {context}

Please provide a structured analysis highlighting any concerning findings and their clinical significance."""
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
    """Generate detailed medical recommendations based on comprehensive analyses"""
    client = openai.OpenAI()
    
    recommendations_prompt = f"""Based on the following diabetic foot analyses, provide evidence-based care recommendations:

Classification Result: {classification_result}

Left Foot Analysis:
{left_analysis}

Right Foot Analysis:
{right_analysis}

Please structure your recommendations in these categories:

1. Urgent Care Requirements:
   - Immediate medical attention needs
   - Critical care procedures
   - Required specialist consultations
   - Emergency warning signs

2. Daily Care Protocol:
   - Detailed cleaning procedures
   - Specific moisturizing recommendations
   - Daily inspection routine
   - Pressure relief techniques
   - Proper drying methods

3. Preventive Measures:
   - Footwear specifications
   - Activity modifications
   - Environmental safety measures
   - Injury prevention strategies
   - Temperature management

4. Monitoring Guidelines:
   - Daily self-examination protocol
   - Critical warning signs
   - When to seek emergency care
   - Follow-up schedule recommendations
   - Progress tracking methods

5. Lifestyle Management:
   - Safe exercise recommendations
   - Dietary considerations
   - Daily activity adaptations
   - Protective measures
   - Risk reduction strategies"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": """You are a senior podiatrist and diabetic care specialist. Provide comprehensive, evidence-based recommendations that are:
1. Specific and actionable
2. Prioritized by urgency
3. Clear and understandable
4. Supported by current medical evidence
5. Focused on preventing complications"""
                },
                {"role": "user", "content": recommendations_prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating recommendations: {str(e)}"

def halaman_2():
    # Page configuration
    st.set_page_config(page_title="Diabetic Foot Analysis System", layout="wide")

    # Custom CSS
    st.markdown("""
        <style>
        .main-title {
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
            color: #2c3e50;
            padding: 1.5rem 0;
            margin-bottom: 2rem;
            background: linear-gradient(120deg, #a1c4fd 0%, #c2e9fb 100%);
            border-radius: 10px;
        }
        
        .section-title {
            font-size: 1.8rem;
            font-weight: bold;
            color: #34495e;
            padding: 1rem 0;
            border-bottom: 2px solid #3498db;
            margin-bottom: 1.5rem;
        }
        
        .info-box {
            background-color: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0 5px 5px 0;
        }
        
        .step-container {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 1.5rem;
        }
        
        .result-card {
            background: #ffffff;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin: 1rem 0;
        }
        
        .highlight-text {
            color: #3498db;
            font-weight: bold;
        }

        .stButton>button {
            background-color: #3498db;
            color: white;
            padding: 0.5rem 2rem;
            font-size: 1.1rem;
            border-radius: 5px;
            border: none;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            transition: all 0.3s;
        }

        .stButton>button:hover {
            background-color: #2980b9;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }

        .upload-section {
            border: 2px dashed #3498db;
            border-radius: 10px;
            padding: 2rem;
            text-align: center;
            margin: 1rem 0;
        }

        .analysis-section {
            background: #f8f9fa;
            padding: 2rem;
            border-radius: 10px;
            margin: 2rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

    # Main Title
    st.markdown('<div class="main-title">🏥 Advanced Diabetic Foot Analysis System</div>', unsafe_allow_html=True)

    # Introduction
    st.markdown("""
        <div class="info-box">
            <h3>📋 About the System</h3>
            <p>Welcome to our advanced diabetic foot analysis system. This platform combines state-of-the-art AI technology 
            with medical expertise to provide comprehensive foot health assessments. The system analyzes your foot images 
            to detect potential diabetic complications and offers personalized care recommendations.</p>
        </div>
    """, unsafe_allow_html=True)

    # Image Upload Instructions
    st.markdown('<div class="section-title">📸 Image Upload Guidelines</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="step-container">
            <h4>For Optimal Results:</h4>
            <ul>
                <li>✅ Ensure good lighting conditions</li>
                <li>✅ Place feet against a plain, light-colored background</li>
                <li>✅ Take photos from a consistent angle</li>
                <li>✅ Include the entire foot in the frame</li>
                <li>✅ Make sure images are clear and in focus</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    # Example Images
    st.markdown('<div class="section-title">📊 Reference Images</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.image("CG010_M_L-rotated1-rotated1.png", caption="Example: Left Foot Position", use_container_width=True)
    with col2:
        st.image("CG010_M_R-rotated1-rotated1.png", caption="Example: Right Foot Position", use_container_width=True)

    # Image Upload Section
    st.markdown('<div class="section-title">📤 Upload Your Images</div>', unsafe_allow_html=True)
    
    upload_col1, upload_col2 = st.columns(2)
    with upload_col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        uploaded_left_image = st.file_uploader("Upload Left Foot Image", type=["jpg", "png", "jpeg"])
        if uploaded_left_image:
            st.image(uploaded_left_image, caption="Left Foot Image", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with upload_col2:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        uploaded_right_image = st.file_uploader("Upload Right Foot Image", type=["jpg", "png", "jpeg"])
        if uploaded_right_image:
            st.image(uploaded_right_image, caption="Right Foot Image", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Analysis Button and Processing
    if uploaded_left_image and uploaded_right_image:
        st.markdown("""
            <div style="text-align: center; padding: 20px;">
                <div style="display: inline-block;">
            </div>
            """, unsafe_allow_html=True)
        
        if st.button("🔍 Start Comprehensive Analysis", key="analyze_button"):
            with st.spinner("🔄 Processing Images... Please wait while we analyze your foot images..."):
                try:
                    # Load model
                    @st.cache_resource
                    def load_model():
                        model = MobileNetV3Model()
                        model.load_state_dict(torch.load('mobilenet_v3_model.pth', weights_only=True), strict=False)
                        model.eval()
                        return model

                    model = load_model()

                    # Process images
                    left_image = Image.open(uploaded_left_image)
                    right_image = Image.open(uploaded_right_image)

                    # Model prediction
                    left_tensor = preprocess_image(left_image).unsqueeze(0)
                    right_tensor = preprocess_image(right_image).unsqueeze(0)

                    with torch.no_grad():
                        prediction = model(left_tensor, right_tensor)
                        is_diabetic = prediction.item() > 0.5

                    prediction_label = "Diabetic" if is_diabetic else "Non-Diabetic"
                    prediction_probability = torch.sigmoid(prediction).item() * 100

                    # Display Results
                    st.markdown('<div class="section-title">🎯 Analysis Results</div>', unsafe_allow_html=True)
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    
                    result_col1, result_col2 = st.columns(2)
                    with result_col1:
                        st.metric(
                            "Classification", 
                            prediction_label,
                            delta="High Confidence" if prediction_probability > 80 else "Moderate Confidence"
                        )
                    with result_col2:
                        st.metric(
                            "Confidence Score", 
                            f"{prediction_probability:.1f}%"
                        )
                    
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Detailed Analysis
                    st.markdown('<div class="section-title">🔬 Detailed Analysis</div>', unsafe_allow_html=True)
                    
                    analysis_col1, analysis_col2 = st.columns(2)
                    with analysis_col1:
                        st.markdown('<div class="result-card">', unsafe_allow_html=True)
                        st.markdown("#### Left Foot Analysis")
                        left_analysis = analyze_image_with_openai(
                            left_image,
                            f"Left foot image. Classification: {prediction_label}"
                        )
                        st.markdown(left_analysis)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with analysis_col2:
                        st.markdown('<div class="result-card">', unsafe_allow_html=True)
                        st.markdown("#### Right Foot Analysis")
                        right_analysis = analyze_image_with_openai(
                            right_image,
                            f"Right foot image. Classification: {prediction_label}"
                        )
                        st.markdown(right_analysis)
                        st.markdown('</div>', unsafe_allow_html=True)

                    # Recommendations
                    st.markdown('<div class="section-title">💡 Personalized Care Recommendations</div>', unsafe_allow_html=True)
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    recommendations = generate_recommendations(
                        prediction_label,
                        left_analysis,
                        right_analysis
                    )
                    st.markdown(recommendations)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Save Results
                    data = {
                        "Date": [str(pd.Timestamp.now())],
                        "Prediction": [prediction_label],
                        "Confidence": [f"{prediction_probability:.1f}%
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close main-container div

def halaman_3():
    # Apply custom CSS for better styling
    st.markdown("""
        <style>
        .dashboard-title {
            text-align: center;
            padding: 20px;
            color: #2c3e50;
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 30px;
            background: linear-gradient(120deg, #84fab0 0%, #8fd3f4 100%);
            border-radius: 10px;
        }
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
        }
        .stat-card:hover {
            transform: translateY(-5px);
        }
        .section-title {
            color: #2c3e50;
            border-left: 5px solid #3498db;
            padding-left: 10px;
            margin: 30px 0 20px 0;
        }
        </style>
    """, unsafe_allow_html=True)

    # Dashboard Title
    st.markdown('<div class="dashboard-title">Analytics Dashboard</div>', unsafe_allow_html=True)

    # Read and validate data
    log_file = "analysis_log.xlsx"
    if not os.path.exists(log_file):
        st.error("⚠️ No analysis data available. Please perform some predictions first.")
        return

    # Load and process data
    df = pd.read_excel(log_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Confidence_Value'] = df['Confidence'].str.rstrip('%').astype(float)

    # Key Metrics Section
    st.markdown('<h2 class="section-title">📊 Key Performance Metrics</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate metrics
    total_predictions = len(df)
    diabetic_percent = (df['Prediction'] == 'Diabetic').mean() * 100
    avg_confidence = df['Confidence_Value'].mean()
    high_risk_cases = ((df['Prediction'] == 'Diabetic') & (df['Confidence_Value'] > 90)).sum()

    # Display metrics with enhanced styling
    metrics = [
        (col1, "Total Predictions", total_predictions, "👥"),
        (col2, "Diabetic Cases", f"{diabetic_percent:.1f}%", "🏥"),
        (col3, "Avg. Confidence", f"{avg_confidence:.1f}%", "📈"),
        (col4, "High Risk Cases", high_risk_cases, "⚠️")
    ]

    for col, title, value, icon in metrics:
        with col:
            st.markdown(f"""
                <div class="stat-card">
                    <h3 style="font-size: 1.1em; color: #7f8c8d;">{title}</h3>
                    <div style="font-size: 1.8em; color: #2c3e50; margin: 10px 0;">{icon} {value}</div>
                </div>
            """, unsafe_allow_html=True)

    # Advanced Visualizations Section
    st.markdown('<h2 class="section-title">📈 Advanced Analytics</h2>', unsafe_allow_html=True)
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Enhanced Prediction Distribution
        fig_donut = {
            "data": [{
                "values": df['Prediction'].value_counts().values,
                "labels": df['Prediction'].value_counts().index,
                "type": "pie",
                "hole": 0.6,
                "marker": {
                    "colors": ["#FF6B6B", "#4ECDC4"],
                    "line": {"color": "#ffffff", "width": 2}
                },
                "textinfo": "label+percent",
                "hoverinfo": "label+value+percent"
            }],
            "layout": {
                "title": "Prediction Distribution",
                "showlegend": True,
                "legend": {"orientation": "h", "y": -0.1},
                "annotations": [{
                    "text": "Distribution<br>Analysis",
                    "showarrow": False,
                    "font": {"size": 14, "color": "#2c3e50"}
                }]
            }
        }
        st.plotly_chart(fig_donut, use_container_width=True)

    with chart_col2:
        # Enhanced Confidence Distribution
        bins = [0, 60, 75, 90, 100]
        labels = ['Low (<60%)', 'Moderate (60-75%)', 'High (75-90%)', 'Very High (>90%)']
        df['Confidence_Range'] = pd.cut(df['Confidence_Value'], bins=bins, labels=labels)
        confidence_dist = df['Confidence_Range'].value_counts().sort_index()
        
        fig_bar = {
            "data": [{
                "x": confidence_dist.index,
                "y": confidence_dist.values,
                "type": "bar",
                "marker": {
                    "color": ["#FFE66D", "#FF9999", "#FF6B6B", "#4ECDC4"],
                    "line": {"color": "#ffffff", "width": 1.5}
                },
                "hovertemplate": "<b>%{x}</b><br>Count: %{y}<extra></extra>"
            }],
            "layout": {
                "title": "Confidence Level Distribution",
                "xaxis": {"title": "Confidence Range", "tickangle": -45},
                "yaxis": {"title": "Number of Cases"},
                "showlegend": False,
                "bargap": 0.2
            }
        }
        st.plotly_chart(fig_bar, use_container_width=True)

    # Time Series Analysis
    st.markdown('<h2 class="section-title">📅 Trend Analysis</h2>', unsafe_allow_html=True)
    
    # Enhanced time series visualization
    df['Date_Only'] = df['Date'].dt.date
    time_series = df.groupby(['Date_Only', 'Prediction']).size().unstack(fill_value=0)
    
    fig_line = {
        "data": [
            {
                "x": time_series.index,
                "y": time_series['Diabetic'] if 'Diabetic' in time_series.columns else [],
                "name": "Diabetic Cases",
                "type": "scatter",
                "mode": "lines+markers",
                "marker": {"color": "#FF6B6B", "size": 8},
                "line": {"width": 3}
            },
            {
                "x": time_series.index,
                "y": time_series['Non-Diabetic'] if 'Non-Diabetic' in time_series.columns else [],
                "name": "Non-Diabetic Cases",
                "type": "scatter",
                "mode": "lines+markers",
                "marker": {"color": "#4ECDC4", "size": 8},
                "line": {"width": 3}
            }
        ],
        "layout": {
            "title": "Daily Prediction Trends",
            "xaxis": {"title": "Date", "gridcolor": "#f0f0f0"},
            "yaxis": {"title": "Number of Cases", "gridcolor": "#f0f0f0"},
            "legend": {"orientation": "h", "y": -0.2},
            "hovermode": "x unified",
            "plot_bgcolor": "white",
            "paper_bgcolor": "white"
        }
    }
    st.plotly_chart(fig_line, use_container_width=True)

    # Interactive Data Table Section
    st.markdown('<h2 class="section-title">📋 Detailed Analysis Records</h2>', unsafe_allow_html=True)
    
    # Date range selector with better styling
    date_col1, date_col2 = st.columns([3, 1])
    with date_col1:
        selected_date_range = st.date_input(
            "Select Date Range",
            value=(df['Date'].min().date(), df['Date'].max().date()),
            min_value=df['Date'].min().date(),
            max_value=df['Date'].max().date()
        )
    
    def calculate_risk_score(row):
        confidence = row['Confidence_Value']
        prediction = row['Prediction']
        
        if prediction == 'Diabetic':
            if confidence >= 90: return ('High Risk', '#FF4444')
            elif confidence >= 75: return ('Moderate Risk', '#FFA000')
            else: return ('Low Risk', '#4CAF50')
        else:
            if confidence >= 90: return ('Minimal Risk', '#2196F3')
            elif confidence >= 75: return ('Low Risk', '#4CAF50')
            else: return ('Follow-up Recommended', '#FFA000')

    if len(selected_date_range) == 2:
        start_date, end_date = selected_date_range
        filtered_df = df[(df['Date'].dt.date >= start_date) & 
                        (df['Date'].dt.date <= end_date)].copy()
        
        # Add risk assessment
        filtered_df['Risk_Info'] = filtered_df.apply(calculate_risk_score, axis=1)
        filtered_df['Risk_Level'] = filtered_df['Risk_Info'].apply(lambda x: x[0])
        filtered_df['Risk_Color'] = filtered_df['Risk_Info'].apply(lambda x: x[1])
        
        # Prepare display dataframe
        display_df = filtered_df[['Date', 'Prediction', 'Confidence_Value', 'Risk_Level']].copy()
        display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        display_df['Confidence_Value'] = display_df['Confidence_Value'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(
            display_df,
            column_config={
                "Date": "Timestamp",
                "Prediction": "Diagnosis",
                "Confidence_Value": "Confidence",
                "Risk_Level": "Risk Assessment"
            },
            hide_index=True
        )
    st.markdown('</div>', unsafe_allow_html=True)  # Close main-container div
# Define the option menu for navigation inside a full-width container
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
            "border-radius": "8px",
            "padding": "10px 15px",
            "box-shadow": "0px 4px 10px rgba(0, 0, 0, 0.15)",
            "transition": "background-color 0.3s ease",
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
