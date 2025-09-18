# AI-Powered Landscape Change Detection

## Overview

This project implements an AI-powered solution for detecting landscape changes using satellite imagery. It leverages a U-Net deep learning model, Gemini AI for analysis, and a Streamlit web interface for user interaction. The application allows users to upload "before" and "after" images, process them using the U-Net model, visualize the detected changes, and generate comprehensive PDF reports.

## Key Features

*   **U-Net Model:** Employs a U-Net deep learning model for accurate image segmentation and change detection.
*   **Gemini AI Integration:** Integrates with Google's Gemini AI for intelligent analysis and insights into detected changes.
*   **Streamlit Interface:** Provides an intuitive web interface for easy interaction and visualization.
*   **Image Preprocessing:** Uses Albumentations for image augmentation and preprocessing to improve model performance.
*   **PDF Report Generation:** Generates detailed PDF reports including before/after images, AI analysis, and evaluation metrics.
*   **Interactive Visualization:** Offers interactive visualizations of the detected changes and model performance metrics.
*   **Multi-Language Support:** Provides multi-language support for AI analysis using the `googletrans` library.

## Technologies Used

*   Python
*   Streamlit
*   PyTorch
*   Albumentations
*   Google Generative AI
*   PIL (Pillow)
*   fpdf
*   streamlit\_image\_comparison
*   streamlit\_drawable\_canvas
*   googletrans
*   imgaug
*   pandas
*   plotly
*   numpy

## Setup and Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    venv\Scripts\activate  # On Windows
    source venv/bin/activate   # On macOS and Linux
    ```

3.  **Install the required packages:**

    ```bash
    pip install streamlit torch albumentations google-generative-ai pillow fpdf streamlit_image_comparison streamlit_drawable_canvas googletrans imgaug pandas plotly numpy
    ```

4.  **Set up Gemini API Key:**

    *   Obtain an API key from Google AI Studio ([https://ai.google.dev/](https://ai.google.dev/)).
    *   Set the API key as an environment variable:

        ```bash
        export GOOGLE_API_KEY="YOUR_API_KEY"
        ```

        (Or set it directly in your code, but this is not recommended for security reasons.)

5.  **Download Pre-trained U-Net Model (Optional):**

    *   Download the pre-trained U-Net model weights (e.g., `unet_model.pth`) and place it in the designated directory (e.g., `models/`).
    *   If you don't have a pre-trained model, you'll need to train your own or find a suitable one online.

6.  **Run the Streamlit application:**

    ```bash
    streamlit run app.py
    ```

## Usage

1.  **Upload Images:** Upload the "before" and "after" satellite images using the Streamlit interface.
2.  **Process Images:** Click the "Process" button to run the change detection analysis.
3.  **View Results:**  The application will display the original images, the detected changes, and AI-powered analysis.
4.  **Generate Report:**  Click the "Generate PDF Report" button to create a comprehensive report of the analysis.

## Directory Structure

```
├── app.py               # Main Streamlit application
├── models/              # Directory for storing pre-trained models
│   └── unet_model.pth  # Example pre-trained U-Net model
├── utils/               # Utility functions and modules
│   ├── image_processing.py # Image preprocessing functions
│   ├── ai_analysis.py    # Gemini AI integration functions
│   └── report_generation.py # PDF report generation functions
├── data/                # Directory for storing example data and results
├── requirements.txt     # List of required Python packages
└── README.md            # This file
```

## Contributing

Contributions are welcome! Please submit a pull request with your proposed changes.

## License

[Specify the License] (e.g., MIT License)
```
