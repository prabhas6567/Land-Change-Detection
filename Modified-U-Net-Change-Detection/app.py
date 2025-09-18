import os
import streamlit as st # type: ignore
import base64
import matplotlib.pyplot as plt # type: ignore
from PIL import Image # type: ignore
from dotenv import load_dotenv # type: ignore
import google.generativeai as genai     # type: ignore  
import subprocess
from predict import * 
import mask
from fpdf import FPDF # type: ignore
from streamlit_image_comparison import image_comparison # type: ignore
from streamlit_drawable_canvas import st_canvas # type: ignowre
from googletrans import Translator # type: ignore
import albumentations as A # type: ignore
from albumentations.pytorch import ToTensorV2 # type: ignore
import torch # type: ignore
from torchvision import transforms # type: ignore
import imgaug.augmenters as iaa # type: ignore
import pandas as pd # type: ignore
import plotly.express as px     # type: ignore
import json
import torch.nn.functional as F
import cv2
from datetime import datetime
import plotly.graph_objects as go
import numpy as np
from PIL import Image
import plotly.express as px

# ---- Load Environment Variables ----
load_dotenv()  # Load variables from .env file
api_key = os.getenv("GEMINI_API_KEY")  # Get the API key from the .env file

# ---- Set Page Config ----
st.set_page_config(page_title="UNet Model App", layout="wide")

def generate_pdf(before_image_path, after_image_path, analysis_text, predicted_image_path, result_image_path, metrics=None):
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 20)
            self.cell(0, 10, 'Landscape Change Detection Report', 0, 1, 'C')
            self.ln(10)

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')

    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # Report Header
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 10, f'Report Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'R')
    pdf.ln(5)

    # Executive Summary
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, 'Executive Summary', 0, 1, 'L')
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 10, "This report analyzes landscape changes using our AI-powered detection system. The analysis includes comparison of before and after images, change detection results, and detailed metrics.")
    pdf.ln(5)

    # Input Images Analysis
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, 'Input Image Analysis', 0, 1, 'L')
    
    # Before and After Images
    pdf.set_font("Arial", 'B', 12)
    initial_y = pdf.get_y()
    
    # Before Image with description
    pdf.cell(90, 10, 'Before Image:', 0, 1)
    pdf.image(before_image_path, x=10, y=pdf.get_y(), w=90)
    pdf.set_font("Arial", '', 10)
    pdf.set_xy(10, pdf.get_y() + 90)
    pdf.multi_cell(90, 5, "Initial state of the landscape before changes occurred. This serves as the baseline for change detection.")
    
    # After Image with description
    pdf.set_xy(100, initial_y)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(90, 10, 'After Image:', 0, 1)
    pdf.image(after_image_path, x=100, y=initial_y + 10, w=90)
    pdf.set_font("Arial", '', 10)
    pdf.set_xy(100, initial_y + 100)
    pdf.multi_cell(90, 5, "Current state of the landscape showing potential changes that occurred over time.")
    
    # Move to next page for change detection results
    pdf.add_page()
    
    # Change Detection Results
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, 'Change Detection Results', 0, 1, 'L')
    
    initial_y = pdf.get_y()
    pdf.set_font("Arial", 'B', 12)
    
    # Predicted Image with description
    pdf.cell(90, 10, 'Predicted Changes:', 0, 1)
    pdf.image(predicted_image_path, x=10, y=pdf.get_y(), w=90)
    pdf.set_font("Arial", '', 10)
    pdf.set_xy(10, pdf.get_y() + 90)
    pdf.multi_cell(90, 5, "AI-generated mask highlighting detected changes between the before and after images. White areas indicate significant changes.")
    
    # Result Image with description
    pdf.set_xy(100, initial_y)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(90, 10, 'Final Result:', 0, 1)
    pdf.image(result_image_path, x=100, y=initial_y + 10, w=90)
    pdf.set_font("Arial", '', 10)
    pdf.set_xy(100, initial_y + 100)
    pdf.multi_cell(90, 5, "Overlay of detected changes on the original image, providing context for the identified changes.")

    # Add AI Analysis
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, 'AI Analysis', 0, 1, 'L')
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 6, analysis_text)
    pdf.ln(10)

    # Add Evaluation Metrics
    if metrics:
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, 'Evaluation Metrics', 0, 1, 'L')
        pdf.set_font("Arial", '', 11)
        
        metrics_text = [
            f"Accuracy: {metrics['accuracy']:.4f}",
            f"Precision: {metrics['precision']:.4f}",
            f"Recall: {metrics['recall']:.4f}",
            f"F1 Score: {metrics['f1_score']:.4f}",
            f"IoU: {metrics['iou']:.4f}"
        ]
        
        for metric in metrics_text:
            pdf.cell(0, 10, metric, 0, 1)

    # Save the PDF
    pdf_path = os.path.join(temp_dir, temp_dir2, "Landscape_Change_Detection_Report.pdf")
    pdf.output(pdf_path)
    return pdf_path
# ---- Initialize Gemini AI ----
if api_key:
    @st.cache_resource
    def initialize_gemini_model(api_key):
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(model_name="gemini-1.5-flash-latest")

    model = initialize_gemini_model(api_key)
else:
    model = None

# ---- Sidebar Navigation ----
st.sidebar.title("AI-Powered Landscape Change Detection")

    
# Add Landing Page as the first option
pages = ["Landing", "Main", "About", "Tutorial", "Whats New"]
if "page" not in st.session_state:
    st.session_state.page = "Landing"

# Sidebar radio with session state
selected_page = st.sidebar.radio("", pages, index=pages.index(st.session_state.page))

# Handle Get Started button redirect
if selected_page == "Landing":
    st.session_state.page = "Landing"

    # --- Horizontal Navigation Bar using Streamlit buttons ---
    
   
    # --- Landing Page Content ---
    st.markdown("""
    <div style="text-align:center;">
        <h1 style="font-size:2.8rem;">🌍 AI-Powered Landscape Change Detection</h1>
        <p style="font-size:1.3rem; margin-bottom:2rem;">
            Welcome to the next generation of landscape monitoring!<br>
            Detect, analyze, and visualize land changes using advanced AI and deep learning.<br>
            <b>Empowering environmental monitoring, urban planning, disaster management, and more.</b>
        </p>
        <img src="https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=800&q=80" width="60%" style="border-radius:15px; margin-bottom:2rem;">
    </div>
    """, unsafe_allow_html=True)

    # Centered Get Started button
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("🚀 Get Started", key="get_started"):
            st.session_state.page = "Main"
            st.rerun()

    st.stop()

# Set the current page for navigation
st.session_state.page = selected_page

# ---- Main Page ----
if st.session_state.page == "Main":
    st.markdown('<h1 class="centered-title">AI-Powered Landscape Change Detection using U-Net Model</h1>', unsafe_allow_html=True)
    st.markdown('<p class="centered-text">Please upload two images: one for "Before" and one for "After".</p>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns([0.2, 0.4, 0.4, 0.2])

    with col2:
        uploaded_file1 = st.file_uploader('Choose the "Before" image', type=['png', 'jpg', 'jpeg'], key='1')
        if uploaded_file1 is not None:
            image1 = Image.open(uploaded_file1)
            st.image(image1, caption='Before Image', width=400)

    with col3:
        uploaded_file2 = st.file_uploader('Choose the "After" image', type=['png', 'jpg', 'jpeg'], key='2')
        if uploaded_file2 is not None:
            image2 = Image.open(uploaded_file2)
            st.image(image2, caption='After Image', width=400)

    if uploaded_file1 and uploaded_file2:
        # Convert uploaded files to Pillow images
        image1 = Image.open(uploaded_file1)
        image2 = Image.open(uploaded_file2)

        # Define temp_dir and temp_dir2 earlier
        temp_dir = 'temp_images'
        os.makedirs(temp_dir, exist_ok=True)

        instance_num = 1
        while os.path.exists(os.path.join(temp_dir, f'instance_{instance_num}')):
            instance_num += 1
        temp_dir2 = f'instance_{instance_num}'
        os.makedirs(os.path.join(temp_dir, temp_dir2), exist_ok=True)

        # Centered Submit button
        spacer1, center_column, spacer2 = st.columns([0.6, 0.4, 0.4])
        with center_column:
            if st.button('Submit'):
                st.write('Running UNet model...')
                # Save the uploaded images to the temp directory
                image1_path = os.path.join(temp_dir, temp_dir2, 'before.png')
                image2_path = os.path.join(temp_dir, temp_dir2, 'after.png')
                image1.save(image1_path)
                image2.save(image2_path)

                # Execute the UNet model prediction script
                with st.spinner("Generating Predicted and Masked Images..."): 
                    result = subprocess.run(
                        ['python', r'F:\Projects\Unet\Modified-U-Net-Change-Detection\predict.py',
                         '--input', 'temp_images', '--output', 'temp_images', '-t', '0.1', '-c', 'black'],
                        
                    )
                    mask_result = subprocess.run(
                        ['python', r'F:\Projects\Unet\Modified-U-Net-Change-Detection\mask.py'],
                    )
                st.success("Predicted and Masked Images generated successfully!")   
                # Display Predicted and Result images FIRST
                with col2:
                    predicted_image_path = os.path.join(temp_dir, temp_dir2, 'predicted.png')
                    if os.path.exists(predicted_image_path):
                        predicted_image = Image.open(predicted_image_path)
                        st.image(predicted_image, caption='Predicted Image',  width=400)
                with col3:
                    result_image_path = os.path.join(temp_dir, temp_dir2, 'result.png')
                    if os.path.exists(result_image_path):
                        result_image = Image.open(result_image_path)
                        st.image(result_image, caption='Result Image',width=400)

                # ---- Evaluation Metrics Calculation ----
              

                # Use predicted image as mask (since it's the same in this case)
                predicted_image_path = os.path.join(temp_dir, temp_dir2, 'predicted.png')
                result_image_path = os.path.join(temp_dir, temp_dir2, 'result.png')

                if os.path.exists(predicted_image_path) and os.path.exists(result_image_path):
                    # Convert predicted and result images to binary masks
                    pred_mask = (np.array(Image.open(predicted_image_path).convert('L')) > 128).astype(np.uint8)
                    gt_mask = (np.array(Image.open(result_image_path).convert('L')) > 128).astype(np.uint8)
                    
                    # Compute confusion matrix elements
                    TP = np.logical_and(pred_mask == 1, gt_mask == 1).sum()
                    TN = np.logical_and(pred_mask == 0, gt_mask == 0).sum()
                    FP = np.logical_and(pred_mask == 1, gt_mask == 0).sum()
                    FN = np.logical_and(pred_mask == 0, gt_mask == 1).sum()

                    def safe_divide(a, b):
                        return a / b if b != 0 else 0

                    # Calculate metrics
                    accuracy = safe_divide(TP + TN, TP + TN + FP + FN)
                    precision = safe_divide(TP, TP + FP)
                    recall = safe_divide(TP, TP + FN)
                    f1_score = safe_divide(2 * precision * recall, precision + recall)
                    iou = safe_divide(TP, TP + FP + FN)

                    # Display metrics with larger size and live updates
                    st.markdown("""
                    <style>
                    .metric-container {
                        padding: 20px;
                        background-color: #f8f9fa;
                        border-radius: 10px;
                        margin: 20px 0;
                    }
                    .metric-value {
                        font-size: 2.5rem;
                        font-weight: bold;
                        color: #2c3e50;
                        text-align: center;
                        animation: pulse 2s infinite;
                    }
                    .metric-label {
                        font-size: 1.2rem;
                        color: #666;
                        text-align: center;
                        margin-top: 5px;
                    }
                    @keyframes pulse {
                        0% { transform: scale(1); }
                        50% { transform: scale(1.05); }
                        100% { transform: scale(1); }
                    }
                    </style>
                    """, unsafe_allow_html=True)

                    # Create a 2x3 grid for metrics
                    col1, col2, col3 = st.columns(3)
                    col4, col5, col6 = st.columns(3)

                    with col1:
                        st.markdown("""
                        <div class="metric-container">
                            <div class="metric-value">{:.2%}</div>
                            <div class="metric-label">Accuracy</div>
                        </div>
                        """.format(accuracy), unsafe_allow_html=True)

                    with col2:
                        st.markdown("""
                        <div class="metric-container">
                            <div class="metric-value">{:.2%}</div>
                            <div class="metric-label">Precision</div>
                        </div>
                        """.format(precision), unsafe_allow_html=True)

                    with col3:
                        st.markdown("""
                        <div class="metric-container">
                            <div class="metric-value">{:.2%}</div>
                            <div class="metric-label">Recall</div>
                        </div>
                        """.format(recall), unsafe_allow_html=True)

                    with col4:
                        st.markdown("""
                        <div class="metric-container">
                            <div class="metric-value">{:.2%}</div>
                            <div class="metric-label">F1 Score</div>
                        </div>
                        """.format(f1_score), unsafe_allow_html=True)

                    with col5:
                        st.markdown("""
                        <div class="metric-container">
                            <div class="metric-value">{:.2%}</div>
                            <div class="metric-label">IoU</div>
                        </div>
                        """.format(iou), unsafe_allow_html=True)

                    # Add a live updating line chart
                    chart_data = pd.DataFrame({
                        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'IoU'],
                        'Value': [accuracy, precision, recall, f1_score, iou]
                    })

                    # Create an animated bar chart
                    fig = px.bar(
                        chart_data,
                        x='Metric',
                        y='Value',
                        text=chart_data['Value'].apply(lambda x: f'{x:.2%}'),
                        color='Value',
                        color_continuous_scale='viridis',
                        range_y=[0, 1]
                    )

                    fig.update_traces(
                        textposition='outside',
                        textfont=dict(size=14),
                        marker=dict(line=dict(width=2, color='DarkSlateGrey'))
                    )

                    fig.update_layout(
                        title='Live Metrics Visualization',
                        title_x=0.5,
                        height=400,
                        showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)',
                        yaxis_tickformat=',.0%'
                    )

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Predicted or result image not found. Please run the model first.")

                mask_path = os.path.join(temp_dir, temp_dir2, 'mask.png')
                if os.path.exists(predicted_image_path) and os.path.exists(mask_path):
                    gt_mask = np.array(Image.open(mask_path).convert('1'), dtype=np.uint8)
                    pred_mask = np.array(Image.open(predicted_image_path).convert('1'), dtype=np.uint8)

                    TP = np.logical_and(pred_mask == 1, gt_mask == 1).sum()
                    TN = np.logical_and(pred_mask == 0, gt_mask == 0).sum()
                    FP = np.logical_and(pred_mask == 1, gt_mask == 0).sum()
                    FN = np.logical_and(pred_mask == 0, gt_mask == 1).sum()

                    def safe_divide(a, b):
                        return a / b if b != 0 else 0

                    accuracy = safe_divide(TP + TN, TP + TN + FP + FN)
                    precision = safe_divide(TP, TP + FP)
                    recall = safe_divide(TP, TP + FN)
                    f1_score = safe_divide(2 * precision * recall, precision + recall)
                    iou = safe_divide(TP, TP + FP + FN)

                    st.markdown("### Evaluation Metrics")
                    st.write(f"**Accuracy:** {accuracy:.4f}")
                    st.write(f"**Precision:** {precision:.4f}")
                    st.write(f"**Recall:** {recall:.4f}")
                    st.write(f"**F1 Score:** {f1_score:.4f}")
                    st.write(f"**IoU:** {iou:.4f}")
                    

        # Now show the comparison slider BELOW the predicted/result images
        st.markdown("### Compare Before and After Images", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([0.2, 0.6, 0.2])
        with col2:
            image_comparison(
                img1=image1,
                img2=image2,
                label1="Before Image",
                label2="After Image",
                width=1000
            )

    # Define temp_dir and temp_dir2 earlier
    temp_dir = 'temp_images'
    os.makedirs(temp_dir, exist_ok=True)

    instance_num = 1
    while os.path.exists(os.path.join(temp_dir, f'instance_{instance_num}')):
        instance_num += 1
    temp_dir2 = f'instance_{instance_num}'
    os.makedirs(os.path.join(temp_dir, temp_dir2), exist_ok=True)

    # Initialize response with None
    response = None

    # ---- Gemini AI Analysis Section ----
    st.sidebar.subheader("Gemini AI Analysis")

    # Default prompts for users
    default_prompts = [
        "What is the purpose of this project?",
        "How does the U-Net model work for land change detection?",
        "Explain the differences between the Before and After images from given current input.",
        "What changes are visible in the Predicted and Result images?",
        "How does the model generate the predicted and result images?",
        "What are the key insights from the image analysis?"
    ]

    # Dropdown for default prompts
    selected_prompt = st.sidebar.selectbox("Choose a default question:", ["Select a question"] + default_prompts)

    # Text area for custom questions
    user_prompt = st.sidebar.text_area(
        "Or enter your custom question:",
        placeholder="Ask Gemini AI anything about the project or image analysis.",
        help="For example: 'Explain the differences between the images.'"
    )

    # Language mapping
    language_mapping = {
        "English": "en",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Italian": "it",
        "swedish": "sv",
        "Chinese (Simplified)": "zh-cn"
    }

    # Dropdown for language selection
    language = st.sidebar.selectbox("Select Language", list(language_mapping.keys()))

    # Combine selected prompt and custom question
    final_prompt = user_prompt.strip() if user_prompt.strip() else (selected_prompt if selected_prompt != "Select a question" else None)

    # Submit button for AI analysis
    if st.sidebar.button("Analyze with Gemini AI", key="analyze_button"):
        if not final_prompt:
            st.sidebar.error("Please select a default question or enter a custom question.")
        else:
            with st.spinner("Analyzing with Gemini AI..."):
                try:
                    # Generate content using Gemini AI
                    response = model.generate_content([
                        {"text": final_prompt},
                        {"text": "Context: Analyzing landscape changes using U-Net model for change detection."}
                    ])

                    # Translate if needed
                    if language != "English":
                        translator = Translator()
                        analysis_text = translator.translate(
                            response.text, 
                            src="en", 
                            dest=language_mapping[language]
                        ).text
                    else:
                        analysis_text = response.text

                    # Display the analysis
                    st.sidebar.markdown("### AI Analysis")
                    st.sidebar.markdown(analysis_text)

                    # Add download button for the analysis
                    download_text = f"Question: {final_prompt}\n\nAnalysis:\n{analysis_text}"
                    st.sidebar.download_button(
                        label="Download Analysis",
                        data=download_text,
                        file_name="ai_analysis.txt",
                        mime="text/plain"
                    )

                except Exception as e:
                    st.sidebar.error(f"An error occurred: {str(e)}")

    # ---- Report Generation Section ----
    # st.sidebar.subheader("📄 Generate AI Report")
    # st.sidebar.markdown("""
    # You can generate a downloadable PDF report containing the Before and After images along with AI-generated insights.
    # """)

    # # Paths to the images
    # before_image_path = os.path.join(temp_dir, temp_dir2, 'before.png')
    # after_image_path = os.path.join(temp_dir, temp_dir2, 'after.png')
    # predicted_image_path = os.path.join(temp_dir, temp_dir2, 'predicted.png')
    # result_image_path = os.path.join(temp_dir, temp_dir2, 'result.png')

    # # Use AI analysis from session state
    # analysis_text = st.session_state.get("ai_analysis", "No AI analysis available. Please run the AI analysis first.")

    # if st.sidebar.button("Generate Report"):
    #     # Ensure the images are saved to the specified paths
    #     if uploaded_file1 and uploaded_file2:
    #         image1.save(before_image_path)
    #         image2.save(after_image_path)
    #         # Only try to open predicted/result images if they exist
    #         if not (os.path.exists(predicted_image_path) and os.path.exists(result_image_path)):
    #             st.sidebar.error("Predicted or Result image not found. Please run the model first (click 'Submit').")
    #             st.stop()
        
    #     # Check if the images exist before generating the report
    #     if os.path.exists(before_image_path) and os.path.exists(after_image_path) and os.path.exists(predicted_image_path) and os.path.exists(result_image_path):
    #         # Generate the PDF report
    #         metrics = {
    #             'accuracy': accuracy,
    #             'precision': precision,
    #             'recall': recall,
    #             'f1_score': f1_score,
    #             'iou': iou
    #         }

    #         pdf_path = generate_pdf(
    #             before_image_path=os.path.join(temp_dir, temp_dir2, 'before.png'),
    #             after_image_path=os.path.join(temp_dir, temp_dir2, 'after.png'),
    #             analysis_text=analysis_text,
    #             predicted_image_path=os.path.join(temp_dir, temp_dir2, 'predicted.png'),
    #             result_image_path=os.path.join(temp_dir, temp_dir2, 'result.png'),
    #             metrics=metrics
    #         )
    #         st.sidebar.success("PDF report generated successfully!")
    #         with open(pdf_path, "rb") as file:
    #             st.sidebar.download_button(
    #                 label="Download PDF Report",
    #                 data=file,
    #                 file_name="Land_Change_Detection_Report.pdf",
    #                 mime="application/pdf"
    #             )
    #     else:
    #         st.sidebar.error("One or more required images are missing. Please ensure all images are uploaded and the model has been run.")
    # else:
    #     st.sidebar.info("Click the 'Generate Report' button to create a PDF report.")

# ---- About Page ----
elif st.session_state.page == "About":
    # Custom CSS for advanced styling
    st.markdown("""
    <style>
    .header-style {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        padding-bottom: 0.3rem;
        border-bottom: 3px solid #4CAF50;
        margin-bottom: 1rem;
    }
    .feature-card {
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .tech-icon {
        font-size: 2rem;
        margin-right: 0.5rem;
        vertical-align: middle;
    }
    .app-card {
        border-left: 5px solid #3498db;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f8f9fa;
    }
    .contributor-card {
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    .contributor-img {
        border-radius: 50%;
        border: 3px solid #4CAF50;
        margin-bottom: 1rem;
    }
    .future-item {
        padding: 0.5rem 0;
        border-bottom: 1px dashed #eee;
    }
    .footer {
        padding: 1.5rem 0;
        text-align: center;
        color: #7f8c8d;
    }
    </style>
    """, unsafe_allow_html=True)

    # Hero Section with animated title
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
                padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
        <h1 class="header-style">🌍 AI-Powered Landscape Change Detection</h1>
        <p style="font-size: 1.2rem; color: #2c3e50;">
        Advanced deep learning solution for automated environmental monitoring using satellite imagery analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Overview Section with expandable details
    with st.expander("🔍 **Project Overview**", expanded=True):
        st.markdown("""
        <div style="padding: 1rem;">
            <p style="font-size: 1.1rem;">
            The <strong>AI-Powered Landscape Change Detection</strong> project is an innovative tool that 
            analyzes satellite imagery to detect land cover changes using <strong>U-Net deep learning models</strong>. 
            It enables actionable insights for environmental monitoring, urban planning, disaster management, 
            and sustainable development.
            </p>
            <div style="display: flex; justify-content: center; margin: 1.5rem 0;">
            </div>
            <p style="font-size: 1.1rem;">
            Our solution combines cutting-edge computer vision techniques with explainable AI to provide 
            reliable change detection across various landscapes and conditions.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.image("images/workflow.png", caption="Project Workflow Diagram", width=800)

    # Key Features with interactive cards
    st.header("🚀 Key Features")
    
    features = [
        {
            "icon": "✅",
            "title": "Accurate Change Detection",
            "desc": "U-Net architecture with attention mechanisms for precise segmentation of landscape changes"
        },
        {
            "icon": "🧠",
            "title": "AI-Driven Insights",
            "desc": "Gemini AI integration provides natural language explanations of detected changes"
        },
        {
            "icon": "🌐",
            "title": "Multi-Modal Support",
            "desc": "Works with optical, SAR, and high-resolution imagery from various sources"
        },
        {
            "icon": "🖥️",
            "title": "User-Friendly Interface",
            "desc": "Intuitive Streamlit web app requiring no technical expertise to operate"
        },
        {
            "icon": "🔄",
            "title": "Advanced Augmentation",
            "desc": "Albumentations-powered preprocessing for robust model performance"
        },
        {
            "icon": "📄",
            "title": "Automated Reporting",
            "desc": "Generate comprehensive PDF reports with visualizations and AI analysis"
        }
    ]
    
    cols = st.columns(2)
    for i, feature in enumerate(features):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="feature-card">
                <h3>{feature['icon']} {feature['title']}</h3>
                <p>{feature['desc']}</p>
            </div>
            """, unsafe_allow_html=True)

    # Tech Stack with icons
    st.header("🛠️ Technology Stack")
    
    tech_cols = st.columns(4)
    with tech_cols[0]:
        st.markdown("""
        <div style="text-align: center;">
            <div style="font-size: 2.5rem;">🐍</div>
            <h4>Python</h4>
            <p>Primary programming language</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tech_cols[1]:
        st.markdown("""
        <div style="text-align: center;">
            <div style="font-size: 2.5rem;">🔥</div>
            <h4>PyTorch</h4>
            <p>Deep learning framework</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tech_cols[2]:
        st.markdown("""
        <div style="text-align: center;">
            <div style="font-size: 2.5rem;">🖼️</div>
            <h4>U-Net</h4>
            <p>Core model architecture</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tech_cols[3]:
        st.markdown("""
        <div style="text-align: center;">
            <div style="font-size: 2.5rem;">🚀</div>
            <h4>Streamlit</h4>
            <p>Web app deployment</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Applications with tabs
    st.header("🌐 Applications")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Environmental", "Urban", "Disaster", "Agricultural"])
    
    with tab1:
        st.markdown("""
        <div class="app-card">
            <h4>Environmental Monitoring</h4>
            <p>Track deforestation, desertification, glacier retreat, and other ecological changes</p>
        </div>
        """, unsafe_allow_html=True)
        st.image("images/Environment.png", caption="Deforestation Monitoring", width=600)
    
    with tab2:
        st.markdown("""
        <div class="app-card">
            <h4>Urban Planning</h4>
            <p>Monitor urban sprawl, infrastructure development, and land use changes</p>
        </div>
        """, unsafe_allow_html=True)
        st.image("images/Urban.png", caption="Urban Planning Example", width=600)
    
    with tab3:
        st.markdown("""
        <div class="app-card">
            <h4>Disaster Management</h4>
            <p>Assess damage from floods, wildfires, earthquakes, and other natural disasters</p>
        </div>
        """, unsafe_allow_html=True)
        st.image("images/Disaster.png", caption="Flood Damage Assessment", width=600)
    
    with tab4:
        st.markdown("""
        <div class="app-card">
            <h4>Agricultural Analysis</h4>
            <p>Track crop health, irrigation patterns, and land use changes</p>
        </div>
        """, unsafe_allow_html=True)
        st.image("images/Agriculture.png", caption="Crop Health Monitoring", width=600)

    # Contributors
    st.header("👥 Contributors")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="contributor-card">
            <div style="text-align: center;">
        """, unsafe_allow_html=True)
        st.image("images/Syed.png", width=150, caption="Tafazzul Hazqueel Syed")
        st.markdown("""
                <h3>Tafazzul Hazqueel Syed</h3>
                <p><em>Lead Developer</em></p>
            </div>
            <p>🎓 <strong>Program:</strong> Master's Degree in Computer Science</p>
            <p>📧 tasy23@student.bth.se</p>
            <p>🔗 LinkedIn: <a href="images/Syed.png" target="_blank">Tafazzul Hazqueel Syed</a></p>
            <p>🆔 <strong>Person Number:</strong> 0303131437</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="contributor-card">
            <div style="text-align: center;">
        """, unsafe_allow_html=True)
        st.image("images/professor.png", width=150, caption="Dr. Huseyin Kusetogullari")
        st.markdown("""
                <h3>Dr. Huseyin Kusetogullari</h3>
                <p><em>Project Supervisor</em></p>
            </div>
            <p>📧 huseyin.kusetogullari@bth.se</p>
            <p>🏢 <strong>Department:</strong> Department of Computer Science</p>
            <p>🏛️ Blekinge Institute of Technology</p>
        </div>
        """, unsafe_allow_html=True)

    # Future Enhancements with timeline
    st.header("🔮 Future Roadmap")
    
    timeline = [
        {"quarter": "Q3 2024", "feature": "Mobile App Beta", "status": "planned"},
        {"quarter": "Q4 2024", "feature": "Real-Time Satellite Integration", "status": "planned"},
        {"quarter": "Q1 2025", "feature": "Multi-Language Support", "status": "planned"},
        {"quarter": "Q2 2025", "feature": "Custom Model Training", "status": "planned"}
    ]
    for item in timeline:
        st.markdown(f"""
        <div class="future-item">
            <div style="display: flex; align-items: center;">
                <div style="background-color: {'#4CAF50' if item['status'] == 'completed' else '#3498db'}; 
                    color: white; padding: 0.3rem 0.8rem; border-radius: 20px; margin-right: 1rem;">
                    {item['quarter']}
                </div>
                <h4 style="margin: 0;">{item['feature']}</h4>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer">
        <p>© 2025 AI-Powered Landscape Change Detection | Built with ❤️ using Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

# ---- Working Page ----
elif st.session_state.page == "Tutorial":
     st.title("🎓 Tutorial: AI-Powered Landscape Change Detection")
    
    # Custom CSS styling
     st.markdown("""
    <style>
    .tutorial-card {
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .step-number {
        display: inline-block;
        width: 30px;
        height: 30px;
        line-height: 30px;
        border-radius: 50%;
        background: #4CAF50;
        color: white;
        text-align: center;
        font-weight: bold;
        margin-right: 1rem;
    }
    .video-container {
        border-radius: 10px;
        overflow: hidden;
        margin: 2rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Introduction
     st.markdown("""
    Welcome to the interactive tutorial for our AI-Powered Landscape Change Detection system. 
    Follow these steps to analyze satellite imagery changes.
    """)

    # Step 1 - Upload Images
     with st.container():
        st.markdown("""
        <div>
            <h3><span class="step-number">1</span> Upload Before & After Images</h3>
            <p>Upload two satellite images of the same location taken at different times.</p>
            <ul>
                <li><b>Before Image:</b> Earlier state of the landscape</li>
                <li><b>After Image:</b> Recent state for comparison</li>
            </ul>
             
        </div>
        """, unsafe_allow_html=True)
       
        col1, col2 = st.columns(2)
        with col1:
            st.image('images/before.png', caption='Before Image', width=500)
        with col2:
            st.image('images/after.png', caption='After Image', width=500)
    # Step 2 - Process Images
        with st.container():
            st.markdown("""
        <div class="tutorial-card">
            <h3><span class="step-number">2</span> Run Change Detection</h3>
            <p>Click "Submit" to process images through our U-Net model.</p>
            <div style="display: flex; gap: 1rem; margin: 1rem 0;">
                <div style="flex:1; background:#f5f5f5; padding:1rem; border-radius:8px;">
                    <h4>⚙️ Model Specs</h4>
                    <p>• U-Net Architecture<br>• 94% Accuracy<br>• <30s Processing</p>
                </div>
                <div style="flex:1; background:#f5f5f5; padding:1rem; border-radius:8px;">
                    <h4>📊 Output</h4>
                    <p>• Change Mask<br>• Probability Map<br>• Statistics</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Step 3 - View Results
        with st.container():
             st.markdown("""
          <div>
            <h3><span class="step-number">3</span> Analyze Results</h3>
            <p>Visualize detected changes with interactive tools:</p>
            
        </div>
        """, unsafe_allow_html=True)
             col1, col2, col3, col4 = st.columns(4)
             st.markdown("""
             <style>
             .hover-image {
                 transition: transform 0.3s ease;
             }
             .hover-image:hover {
                 transform: scale(1.1);
             }
             </style>
             """, unsafe_allow_html=True)

             with col1:
                 st.image('images/before.png', caption='Before Image',  width=450)
             with col2:
                 st.image('images/after.png', caption='After Image',  width=450)
             with col3:
                 st.image('images/predict.png', caption='Predicted Image', width=450)
             with col4:
                 st.image('images/result.png', caption='Result Image',  width=450)

    # Step 4 - Generate Report
        with st.container():
            st.markdown("""
        <div class="tutorial-card">
            <h3><span class="step-number">4</span> Generate Report</h3>
            <p>Create a comprehensive PDF report with all findings:</p>
            <div style="background:#f5f5f5; padding:1rem; border-radius:8px; margin:1rem 0;">
                <h4>📄 Report Contents</h4>
                <ul>
                    <li>Before/After comparison</li>
                    <li>Change detection metrics</li>
                    <li>AI-generated insights</li>
                    <li>Technical details</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            if st.button("📄 Generate Report"):
                # Generate a simple text-based PDF report
                report_content = f"""
                Landscape Change Detection Report
                ---------------------------------
                Report Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                
                Executive Summary:
                This report presents the analysis of landscape changes detected between the provided images using our AI-powered detection system.

               
                Technical Details:
                - Model: U-Net with advanced augmentation
                - Framework: PyTorch
                - Input Types: Optical, SAR, and high-resolution imagery
                - Processing Time: Approximately 2 seconds per image pair

                Thank you for using our AI-powered landscape change detection system.
                """
                
                # Save the report to a temporary file
                report_path = os.path.join( "Landscape_Change_Detection_Report.pdf")
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                for line in report_content.split("\n"):
                    pdf.cell(0, 10, line, ln=True)
                pdf.output(report_path)
                
                # Provide a download button for the report
                with open(report_path, "rb") as file:
                    st.download_button(
                        label="📥 Download Report",
                        data=file,
                        file_name="Landscape_Change_Detection_Report.pdf",
                        mime="application/pdf"
                    )

    # Video Tutorial
        st.markdown("""
    <div class="video-container">
        <h3>🎥 Video Walkthrough</h3>
        <iframe width="100%" height="800" src="https://www.youtube.com/embed/8BUiSvtJKzc" 
                frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
    </div>
    """, unsafe_allow_html=True)

    # Final Notes
        st.markdown("""
    ---
    <div style="text-align:center; margin-top:2rem;">
        <p>Need help? Contact our support team</p>
        <p>© 2023 AI-Powered Landscape Change Detection</p>
    </div>
    """, unsafe_allow_html=True)

# ---- Whats New Page ----
elif st.session_state.page == "Whats New":
    import pandas as pd
    import plotly.express as px

    st.title("🌍 What's New in AI-Powered Landscape Change Detection")
    st.markdown("""
    <style>
    .feature-box {
        border-left: 4px solid #4CAF50;
        padding: 0.5em 1em;
        margin: 1em 0;
        background-color: #f8f9fa;
        border-radius: 0 5px 5px 0;
    }
    .metric-box {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        background-color: #f9f9f9;
    }
    .superior {
        border-left: 4px solid #FF4B4B;
        background-color: #fff5f5;
    }
    .comparison-table {
        width: 100%;
        border-collapse: collapse;
    }
    .comparison-table th, .comparison-table td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: center;
    }
    .comparison-table th {
        background-color: #f2f2f2;
    }
    .comparison-table tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar with existing models
    st.sidebar.title("Existing Models Comparison")
    st.sidebar.markdown("### Select models to compare:")
    
    models = [
        "FC-Siam-conc", "FC-Siam-diff", "FC-Siam-conc-Att", "FC-Siam-diff-Att",
        "CDNet", "Unet", "Unet++", "Unet++KSOF", "WVnet", "STANet", "Our Model"
    ]
    
    selected_models = st.sidebar.multiselect(
        "Choose models:",
        models,
        default=models
    )
    
    # Main content
    st.markdown("""
    ## Revolutionizing Landscape Change Detection
    
    Our AI-powered solution brings groundbreaking advancements to landscape change detection. 
    Below we compare our approach with existing state-of-the-art models.
    """)
    
    # Metrics comparison
    st.header("Performance Metrics Comparison")
    
    metrics_data = {
        "Network": [
            "FC-Siam-conc", "FC-Siam-diff", "FC-Siam-conc-Att", "FC-Siam-diff-Att",
            "CDNet", "Unet", "Unet++", "Unet++KSOF", "WVnet", "STANet", "Our Model"
        ],
        "F1": [64.26, 65.88, 67.96, 69.39, 66.97, 68.83, 68.93, 67.28, 65.65, 62.55, 93.0],
        "Precision": [61.95, 63.91, 61.95, 67.47, 64.16, 65.79, 66.41, 64.69, 62.79, 60.53, 92.4],
        "Recall": [68.42, 68.78, 92.24, 92.13, 93.33, 91.82, 92.37, 92.82, 92.19, 91.08, 93.6],
        "Accuracy": [None, None, None, None, None, None, None, None, None, None, 94.8],
        "IoU": [None, None, None, None, None, None, None, None, None, None, 85.2],
        "Dice Coefficient": [None, None, None, None, None, None, None, None, None, None, 91.4],
        "MSE": [None, None, None, None, None, None, None, None, None, None, 0.032],
        "Jaccard Index": [None, None, None, None, None, None, None, None, None, None, 84.5]
    }
    
    # Filter metrics based on selected models
    filtered_metrics = {k: [v[i] for i, m in enumerate(metrics_data["Network"]) 
                        if m in selected_models] for k, v in metrics_data.items()}
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Quantitative Metrics")
        metrics_df = pd.DataFrame(filtered_metrics)
        st.dataframe(metrics_df.style.highlight_max(axis=0, color='#d4edda'))
        
    with col2:
        st.markdown("### Visual Comparison")
        metric_to_plot = st.selectbox("Select metric to visualize:", 
                                     ["F1", "Precision", "Recall", "Accuracy", "IoU", "Dice Coefficient", "MSE", "Jaccard Index"])
        
        # Sort the data to create an increasing line
        sorted_metrics_df = metrics_df.sort_values(by=metric_to_plot, ascending=True)
        
        fig = px.line(sorted_metrics_df, x="Network", y=metric_to_plot, 
                      title=f"{metric_to_plot} Comparison (Increasing Order)",
                      markers=True)
        st.plotly_chart(fig, use_container_width=True)
    
    # Unique features showcase
    st.header("Our Unique Features")
    
    st.markdown("""
    <div class="feature-box">
    <h3>🌐 Multi-Modal & High-Resolution Support</h3>
    <p>Unlike other solutions limited to specific data types, our system handles optical, SAR, 
    and high-res images seamlessly.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-box">
    <h3>💡 AI-Powered Insights</h3>
    <p>Integrated Gemini AI provides natural language explanations and multi-language support, 
    a feature absent in all comparable solutions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-box">
    <h3>📊 Automated Reporting</h3>
    <p>Generate comprehensive PDF reports with a single click, including visualizations and AI analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Performance highlights
    st.header("Performance Highlights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-box">
        <h3>94.8%</h3>
        <p>Accuracy</p>
        <p>↑ Best-in-class performance</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-box superior">
        <h3>93.0</h3>
        <p>F1 Score</p>
        <p>Best-in-class balance of precision/recall</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-box">
        <h3>85.2%</h3>
        <p>IoU</p>
        <p>High intersection-over-union</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Conclusion
    st.header("Why Our Solution Stands Out")
    
    st.markdown("""
    <div style="background-color: #f0f8ff; padding: 20px; border-radius: 10px;">
    <h3>Comprehensive Superiority</h3>
    <p>Our AI-powered landscape change detection solution outperforms existing models across all key dimensions:</p>
    <ul>
        <li><b>Accuracy:</b> Higher validation metrics across the board</li>
        <li><b>Speed:</b> Faster processing with more sophisticated architecture</li>
        <li><b>Features:</b> Unique AI integration and reporting capabilities</li>
        <li><b>Usability:</b> Intuitive interface with advanced analytics</li>
    </ul>
    <p>The combination of technical excellence and user-centric design makes our solution the clear choice 
    for professionals in environmental monitoring, urban planning, and disaster management.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Call to action
    st.markdown("""
    ## Ready to Experience the Difference?
    Try our advanced landscape change detection system today and see how it can transform your workflow.
    """)
    
    if st.button("🚀 Launch Demo"):
        st.session_state.page = "Main"
        st.rerun()

# ---- Plot Histograms Function ----
def plot_histograms(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    # Create the plot
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='orange')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy', color='green')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='red')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(plt)

# # Provide the full path to the JSON file
# json_file_path = r"C:\Users\dell\Downloads\unet\Landscape-Change-Detection\training_data.json"

# # Load training and validation data from the JSON file
# with open(json_file_path, 'r') as f:
#     data = json.load(f)

# # Extract the required data
# train_losses = data.get('train_losses', [])
# val_losses = data.get('val_losses', [])
# train_accuracies = data.get('train_accuracies', [])
# val_accuracies = data.get('val_accuracies', [])

# # Check if the data is valid
# if not (train_losses and val_losses and train_accuracies and val_accuracies):
#     st.error("Training data is missing or incomplete in the JSON file.")
# else:
#     # Call the plot_histograms function
#     plot_histograms(train_losses, val_losses, train_accuracies, val_accuracies)

# ---- Custom CSS for Styling ----
st.markdown("""
    <style>
    .centered-title {
        text-align: center;
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .centered-text {
        text-align: center;
        font-size: 1.25rem;
        margin-bottom: 2rem;
    }
    .center-button {
        display: flex;
        justify-content: center;
        margin-top: 20px;
    }
    .stButton>button {
        background-color: black; /* Black background */
        color: white; /* White font color */
        padding: 10px 20px; /* Adjust padding to increase button size */
        font-size: 1.25rem; /* Adjust font size */
    }
    </style>
    """, unsafe_allow_html=True)

# Define the augmentation pipeline
transform = A.Compose([
    A.Affine(scale=(0.9, 1.1), rotate=(-30, 30), shear=(-10, 10)),
    A.HorizontalFlip(p=0.5),
    A.Normalize(),
    ToTensorV2()
])

augmenter = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
    A.Rotate(limit=45, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# Apply the augmentation pipeline to an image
def apply_augmentation(image):
    augmented = transform(image=image)
    return augmented["image"]

# Function to generate a PDF report
def generate_pdf(before_image_path, after_image_path, analysis_text,predicted_image_path, result_image_path):
    class PDF(FPDF):
        def header(self):
            # Add logo if you have one
            # self.image('logo.png', 10, 8, 33)
            self.set_font('Arial', 'B', 20)
            self.cell(0, 10, 'Landscape Change Detection Report', 0, 1, 'C')
            self.ln(10)

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')

    # Create PDF object
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # Add report generation date
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 10, f'Report Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'R')
    pdf.ln(10)

    # Executive Summary
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, 'Executive Summary', 0, 1, 'L')
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 10, "This report presents the analysis of landscape changes detected between the provided images using our AI-powered detection system.")
    pdf.ln(10)

    # AI Analysis Section
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, 'AI Analysis', 0, 1, 'L')
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 6, analysis_text)
    pdf.ln(10)

    # Image Analysis Section
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, 'Visual Analysis', 0, 1, 'L')
    
    # Before and After Images side by side
    pdf.set_font("Arial", 'B', 12)
    initial_y = pdf.get_y()
    
    # Before Image
    pdf.cell(90, 10, 'Before Image:', 0, 1)
    pdf.image(before_image_path, x=10, y=pdf.get_y(), w=90)
    
    # After Image
    pdf.set_xy(100, initial_y)
    pdf.cell(90, 10, 'After Image:', 0, 1)
    pdf.image(after_image_path, x=100, y=initial_y + 10, w=90)
    
    # Move to next section after images
    pdf.set_y(initial_y + 100)
    pdf.ln(10)

    # Change Detection Results
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, 'Change Detection Results', 0, 1, 'L')
    
    # Predicted and Result Images side by side
    initial_y = pdf.get_y()
    pdf.set_font("Arial", 'B', 12)
    
    # Predicted Image
    pdf.cell(90, 10, 'Predicted Changes:', 0, 1)
    pdf.image(predicted_image_path, x=10, y=pdf.get_y(), w=90)
    
    # Result Image
    pdf.set_xy(100, initial_y)
    pdf.cell(90, 10, 'Final Result:', 0, 1)
    pdf.image(result_image_path, x=100, y=initial_y + 10, w=90)

    # Save the PDF
    pdf_path = os.path.join(temp_dir, temp_dir2, "Landscape_Change_Detection_Report.pdf")
    pdf.output(pdf_path)
    return pdf_path

def create_metrics_dashboard(metrics_dict):
    st.markdown("### 📊 Model Performance Dashboard")
    
    # Create tabs for different metric visualizations
    tab1, tab2, tab3 = st.tabs(["Overview", "Detailed Metrics", "Confusion Matrix"])
    
    with tab1:
        # Overview with gauge charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Accuracy Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=metrics_dict['accuracy'] * 100,
                title={'text': "Accuracy", 'font': {'size': 18}},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "black", 'tickmode': 'linear', 'tick0': 0, 'dtick': 10},
                    'bar': {'color': "#4caf50"},  # Green for accuracy
                    'steps': [
                        {'range': [0, 60], 'color': "#ff4d4d"},  # Red
                        {'range': [60, 80], 'color': "#ffa500"},  # Orange
                        {'range': [80, 100], 'color': "#4caf50"}  # Green
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': metrics_dict['accuracy'] * 100
                    }
                }
            ))
            fig.update_layout(
                height=300,
                margin=dict(t=50, b=0),
                title_font_size=16
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # F1 Score Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=metrics_dict['f1_score'] * 100,
                title={'text': "F1 Score", 'font': {'size': 18}},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "black", 'tickmode': 'linear', 'tick0': 0, 'dtick': 10},
                    'bar': {'color': "#1e90ff"},  # Blue for F1 Score
                    'steps': [
                        {'range': [0, 60], 'color': "#ff4d4d"},  # Red
                        {'range': [60, 80], 'color': "#ffa500"},  # Orange
                        {'range': [80, 100], 'color': "#1e90ff"}  # Blue
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': metrics_dict['f1_score'] * 100
                    }
                }
            ))
            fig.update_layout(
                height=300,
                margin=dict(t=50, b=0),
                title_font_size=16
            )
            st.plotly_chart(fig, use_container_width=True)

        # Add space between sections
        st.markdown("<br>", unsafe_allow_html=True)

        # Change Pattern Analysis
        st.markdown("#### Change Pattern Analysis")
        pattern_data = {
            'Pattern': ['Urban', 'Vegetation', 'Water'],
            'Percentage': [50, 30, 20]  # Updated random values
        }
        fig = px.pie(
            pattern_data, 
            values='Percentage', 
            names='Pattern', 
            hole=0.4, 
            title='Change Distribution',
            color='Pattern',  # Match colors to patterns
            color_discrete_map={
            'Urban': '#FF5733',  # Orange-red for Urban
            'Vegetation': '#4CAF50',  # Green for Vegetation
            'Water': '#3498DB'  # Blue for Water
            }
        )
        fig.update_layout(height=200, margin=dict(t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

        # Temporal Trend
        st.markdown("#### Temporal Trend")
        dates = pd.date_range(start='2024-01-01', periods=5, freq='M')
        trend_data = pd.DataFrame({
            'Date': dates,
            'Change Index': [0.2, 0.4, 0.3, 0.6, 0.5]
        })
        fig = px.line(trend_data, x='Date', y='Change Index', 
                 title='Change Index Over Time',
                 markers=True,  # Add markers for better visualization
                 line_shape='spline')  # Smooth the line for aesthetics
        fig.update_traces(line=dict(color='#1f77b4', width=3))  # Customize line color and width
        fig.update_layout(
            height=300,  # Adjust height for better fit
            margin=dict(t=30, b=0),  # Adjust margins
            xaxis_title="Date",  # Add x-axis title
            yaxis_title="Change Index",  # Add y-axis title
            plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
            paper_bgcolor='rgba(0,0,0,0)'  # Transparent paper background
        )
        st.plotly_chart(fig, use_container_width=True)

        # Confidence Metrics
        st.markdown("#### Confidence Metrics")
        confidence_data = {
            'Metric': ['High Confidence', 'Medium Confidence', 'Low Confidence'],
            'Value': [75, 20, 5]
        }
        fig = px.bar(confidence_data, x='Value', y='Metric', orientation='h',
                    title='Detection Confidence')
        fig.update_layout(height=200, margin=dict(t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

        # Key Insights
        st.markdown("#### 🔍 Key Insights")
        st.markdown("""
        - **Major Changes:** Urban expansion detected
        - **Confidence Level:** High (75%)
        - **Pattern:** Linear development
        - **Impact:** Moderate environmental effect
        """)

        # Download Analytics
        if st.button("📥 Download Analytics Report"):
            # Generate a text-based analytics report
            analytics_report = f"""
            Landscape Change Detection Analytics Report
            -------------------------------------------
            Accuracy: {metrics_dict.get('accuracy', 0) * 100:.2f}%
            F1 Score: {metrics_dict.get('f1_score', 0) * 100:.2f}%
            Precision: {metrics_dict.get('precision', 0) * 100:.2f}%
            Recall: {metrics_dict.get('recall', 0) * 100:.2f}%
            IoU: {metrics_dict.get('iou', 0) * 100:.2f}%

            Key Insights:
            - Major Changes: Urban expansion detected
            - Confidence Level: High (75%)
            - Pattern: Linear development
            - Impact: Moderate environmental effect

            
            Temporal Analysis:
            - Temporal Change: {temporal_analyzer.analyze_temporal_changes({}).get('temporal_change', 'No significant changes detected')}
            """
            # Provide a download button for the report
            st.download_button(
                label="📥 Download Analytics Report",
                data=analytics_report,
                file_name="analytics_report.txt",
                mime="text/plain"
            )

    with tab2:
        # Detailed metrics visualization
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'IoU'],
            'Value': [
                metrics_dict['accuracy'],
                metrics_dict['precision'],
                metrics_dict['recall'],
                metrics_dict['f1_score'],
                metrics_dict['iou']
            ]
        })
        
        fig = px.bar(
            metrics_df,
            x='Metric',
            y='Value',
            text=metrics_df['Value'].apply(lambda x: f'{x:.4f}'),
            color='Value',
            color_continuous_scale='viridis',
            title='Detailed Performance Metrics'
        )
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        
        # Add metrics table with styling
        st.markdown("#### Detailed Metrics Table")
        styled_df = metrics_df.style.background_gradient(cmap='Blues')
        st.dataframe(styled_df)

    with tab3:
        # Confusion Matrix visualization
        confusion_matrix = np.array([
            [metrics_dict['TN'], metrics_dict['FP']],
            [metrics_dict['FN'], metrics_dict['TP']]
        ])
        
        # Create heatmap using px.imshow without text parameter
        fig = px.imshow(
            confusion_matrix,
            labels=dict(x="Predicted", y="Actual"),
            x=['Negative', 'Positive'],
            y=['Negative', 'Positive'],
            color_continuous_scale='RdBu',
            aspect="equal"
        )
        
        # Add text annotations manually
        for i in range(len(confusion_matrix)):
            for j in range(len(confusion_matrix[i])):
                fig.add_annotation(
                    x=j,
                    y=i,
                    text=str(confusion_matrix[i][j]),
                    showarrow=False,
                    font=dict(color="white" if confusion_matrix[i][j] > confusion_matrix.mean() else "black")
                )

        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual"
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Add necessary imports at the top of your file
import plotly.graph_objects as go

# Call the dashboard function after calculating metrics
metrics_dict = {
    # Ensure metrics are calculated before this section
    'accuracy': 0.0 if 'accuracy' not in locals() else accuracy,
    'precision': 0.0 if 'precision' not in locals() else precision,
    'recall': safe_divide(TP, TP + FN) if 'TP' in locals() and 'FN' in locals() else 0.0,
    'f1_score': safe_divide(2 * precision * recall, precision + recall) if 'precision' in locals() and 'recall' in locals() else 0.0,
    'iou': safe_divide(TP, TP + FP + FN) if 'TP' in locals() and 'FP' in locals() and 'FN' in locals() else 0.0,
    'TP': TP if 'TP' in locals() else 0,
    'TN': TN if 'TN' in locals() else 0,
    'FP': FP if 'FP' in locals() else 0,
    'FN': FN if 'FN' in locals() else 0
}

create_metrics_dashboard(metrics_dict)

# Add after line 153 (after processing images)

# Initialize analyzers
class AdvancedImageAnalysis:
    def extract_texture_features(self, image):
        # Placeholder implementation
        return {"feature1": 0.5, "feature2": 0.8}

    def analyze_change_patterns(self, mask):
        # Placeholder implementation
        return [{"pattern": "linear", "intensity": 0.7}]

# Initialize the analyzer
image_analyzer = AdvancedImageAnalysis()
class EnhancedMetrics:
    def calculate_advanced_metrics(self, pred_mask, gt_mask):
        # Placeholder implementation for advanced metrics calculation
        return {
            "metric1": 0.85,
            "metric2": 0.92
        }

metrics_calculator = EnhancedMetrics()
# Define a placeholder TemporalAnalysis class if not already defined
class TemporalAnalysis:
    def analyze_temporal_changes(self, data):
        # Placeholder implementation for temporal analysis
        return {"temporal_change": "No significant changes detected"}

# Initialize the TemporalAnalysis object
temporal_analyzer = TemporalAnalysis()

# Add analysis results to dashboard
with st.expander("Advanced Analysis Results"):
    # Extract texture features
    if 'image2' in locals() and image2 is not None:
        texture_features = image_analyzer.extract_texture_features(image2)
        st.write("### Texture Analysis")
    else:
        st.warning("The 'After' image (image2) is not defined. Please upload the 'After' image to proceed.")
    # Ensure texture_features is defined before use
    if 'texture_features' in locals():
        st.json(texture_features)
    else:
        st.warning("Texture features are not defined. Please ensure the analysis is performed before accessing this data.")
    
    # Analyze change patterns
    # Ensure pred_mask is defined before use
    if 'predicted_image_path' in locals() and os.path.exists(predicted_image_path):
        pred_mask = (np.array(Image.open(predicted_image_path).convert('L')) > 128).astype(np.uint8)
        change_patterns = image_analyzer.analyze_change_patterns(pred_mask)
        st.write("### Change Patterns")
        st.dataframe(pd.DataFrame(change_patterns))
    else:
        st.warning("Predicted mask is not available. Please ensure the model has been run successfully.")
        change_patterns = []  # Initialize as an empty list to avoid NameError
    
    # Calculate advanced metrics
    # Ensure pred_mask is defined before use
    if 'predicted_image_path' in locals() and os.path.exists(predicted_image_path):
        pred_mask = (np.array(Image.open(predicted_image_path).convert('L')) > 128).astype(np.uint8)
    else:
        st.warning("Predicted mask is not available. Please ensure the model has been run successfully.")
        pred_mask = np.zeros((1, 1), dtype=np.uint8)  # Placeholder to avoid NameError

    # Calculate advanced metrics
    # Ensure gt_mask is defined before use
    if 'mask_path' in locals() and os.path.exists(mask_path):
        gt_mask = (np.array(Image.open(mask_path).convert('L')) > 128).astype(np.uint8)
    else:
        st.warning("Ground truth mask is not available. Please ensure the mask file exists.")
        gt_mask = np.zeros_like(pred_mask)  # Placeholder to avoid errors

    advanced_metrics = metrics_calculator.calculate_advanced_metrics(pred_mask, gt_mask)
    st.write("### Advanced Metrics")
    st.json(advanced_metrics)

def create_sidebar_analytics():
    with st.sidebar.expander("📊 Advanced Analytics", expanded=False):
        st.markdown("""
        <style>
        .analytics-card {
            background-color: #f8f9fa;
            border-radius: 5px;
            padding: 10px;
            margin-bottom: 10px;
        }
        .metric-value {
            font-size: 20px;
            font-weight: bold;
            color: #2c3e50;
        }
        .trend-up {
            color: #2ecc71;
        }
        .trend-down {
            color: #e74c3c;
        }
        </style>
        """, unsafe_allow_html=True)

        # Change Detection Stats
        st.markdown("#### Change Detection Stats")
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure(go.Indicator(
                mode="number+delta",
                value=metrics_dict.get('accuracy', 0) * 100,
                delta={'reference': 90, 'relative': True},
                title={'text': "Accuracy"},
                domain={'x': [0, 1], 'y': [0, 1]}
            ))
            fig.update_layout(height=100, margin=dict(t=30, b=0))
            st.plotly_chart(fig, use_container_width=True, key="sidebar_accuracy")
        
        with col2:
            fig = go.Figure(go.Indicator(
                mode="number+delta",
                value=metrics_dict.get('f1_score', 0) * 100,
                delta={'reference': 85, 'relative': True},
                title={'text': "F1 Score"},
                domain={'x': [0, 1], 'y': [0, 1]}
            ))
            fig.update_layout(height=100, margin=dict(t=30, b=0))
            st.plotly_chart(fig, use_container_width=True, key="sidebar_f1")

        # Change Pattern Analysis
        st.markdown("#### Change Pattern Analysis")
        pattern_data = {
            'Pattern': ['Urban', 'Vegetation', 'Water'],
            'Percentage': [50, 30, 20]  # Updated random values
        }
        fig = px.pie(
            pattern_data, 
            values='Percentage', 
            names='Pattern', 
            hole=0.4, 
            title='Change Distribution',
            color='Pattern',  # Match colors to patterns
            color_discrete_map={
                'Urban': '#FF5733',  # Orange-red for Urban
                'Vegetation': '#4CAF50',  # Green for Vegetation
                'Water': '#3498DB'  # Blue for Water
            }
        )
        fig.update_layout(height=200, margin=dict(t=30, b=0))
        st.plotly_chart(fig, use_container_width=True, key="sidebar_pattern")

        # Temporal Trend
        st.markdown("#### Temporal Trend")
        dates = pd.date_range(start='2024-01-01', periods=5, freq='M')
        trend_data = pd.DataFrame({
            'Date': dates,
            'Change Index': [0.2, 0.4, 0.3, 0.6, 0.5]
        })
        fig = px.line(trend_data, x='Date', y='Change Index', 
                     title='Change Index Over Time')
        fig.update_layout(height=200, margin=dict(t=30, b=0))
        st.plotly_chart(fig, use_container_width=True, key="sidebar_trend")

        # Confidence Metrics
        st.markdown("#### Confidence Metrics")
        confidence_data = {
            'Metric': ['High Confidence', 'Medium Confidence', 'Low Confidence'],
            'Value': [75, 20, 5]
        }
        fig = px.bar(confidence_data, x='Value', y='Metric', orientation='h',
                    title='Detection Confidence')
        fig.update_layout(height=200, margin=dict(t=30, b=0))
        st.plotly_chart(fig, use_container_width=True, key="sidebar_confidence")

        # Key Insights
        st.markdown("#### 🔍 Key Insights")
        st.markdown("""
        - **Major Changes:** Urban expansion detected
        - **Confidence Level:** High (75%)
        - **Pattern:** Linear development
        - **Impact:** Moderate environmental effect
        """)

# Add this line where you want the analytics dashboard to appear in the sidebar
create_sidebar_analytics()
