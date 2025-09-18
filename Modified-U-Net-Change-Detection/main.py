import streamlit as st
import torch
import os
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time

from unet import BasicUNet

def load_model():
    model = BasicUNet(in_channels=6, out_channels=1)
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    return model

def get_oscd_locations():
    # Path to OSCD dataset
    oscd_path = os.path.join(os.path.dirname(__file__), 'OSCD')
    images_path = os.path.join(oscd_path, 'images', 'Onera Satellite Change Detection dataset - Images')
    locations = [d for d in os.listdir(images_path) if os.path.isdir(os.path.join(images_path, d))]
    return locations, images_path

def load_image_pairs(location, images_path):
    pair_path = os.path.join(images_path, location, 'pair')
    image_files = sorted([f for f in os.listdir(pair_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    if len(image_files) >= 2:
        img1 = Image.open(os.path.join(pair_path, image_files[0])).convert('RGB')
        img2 = Image.open(os.path.join(pair_path, image_files[1])).convert('RGB')
        return img1, img2
    return None, None

def load_mask(location):
    oscd_path = os.path.join(os.path.dirname(__file__), 'OSCD')
    mask_path = os.path.join(oscd_path, 'train_labels', 
                            'Onera Satellite Change Detection dataset - Train Labels',
                            location, 'cm', 'cm.png')
    if os.path.exists(mask_path):
        return Image.open(mask_path).convert('L')
    return None

def main():
    st.title("Satellite Image Change Detection")
    
    # Add file uploaders for images
    col1, col2 = st.columns(2)
    with col1:
        uploaded_img1 = st.file_uploader("Upload Time 1 Image", type=['png', 'jpg', 'jpeg'])
        if uploaded_img1:
            img1 = Image.open(uploaded_img1).convert('RGB')
            st.image(img1, caption="Time 1 Image", use_column_width=True)
    
    with col2:
        uploaded_img2 = st.file_uploader("Upload Time 2 Image", type=['png', 'jpg', 'jpeg'])
        if uploaded_img2:
            img2 = Image.open(uploaded_img2).convert('RGB')
            st.image(img2, caption="Time 2 Image", use_column_width=True)
    
    # Add submit button
    if uploaded_img1 and uploaded_img2:
        if st.button("Detect Changes", type="primary"):
            # Show processing message
            with st.spinner('Processing images... Please wait.'):
                # Add artificial delay to show processing
                time.sleep(2)
                
                # Create figure with 3x2 subplots
                fig, axes = plt.subplots(3, 2, figsize=(15, 18))
                fig.suptitle('Change Detection Analysis', fontsize=16)
                
                # Show progress message
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Row 1: Original images
                status_text.text("Loading images...")
                progress_bar.progress(20)
                
                axes[0, 0].imshow(np.array(img1))
                axes[0, 0].set_title('Time 1 Image')
                axes[0, 0].axis('off')
                
                axes[0, 1].imshow(np.array(img2))
                axes[0, 1].set_title('Time 2 Image')
                axes[0, 1].axis('off')
                
                # Load and run model
                status_text.text("Running change detection model...")
                progress_bar.progress(40)
                
                model = load_model()
                
                # Preprocess images
                transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
                ])
                
                # Transform images
                img1_tensor = transform(img1).unsqueeze(0)
                img2_tensor = transform(img2).unsqueeze(0)
                
                # Concatenate images
                input_tensor = torch.cat([img1_tensor, img2_tensor], dim=1)
                
                # Get prediction
                with torch.no_grad():
                    pred = model(input_tensor)
                    pred = torch.sigmoid(pred)
                
                # Convert prediction to numpy
                pred_np = pred[0, 0].cpu().numpy()
                pred_mask = (pred_np > 0.5).astype(float)
                
                # Update progress
                status_text.text("Generating visualizations...")
                progress_bar.progress(60)
                
                # Row 2: Images with overlays
                # Time 1 + Prediction overlay
                axes[1, 0].imshow(np.array(img1))
                axes[1, 0].imshow(pred_mask, cmap='Reds', alpha=0.4)
                axes[1, 0].set_title('Ground Truth')
                axes[1, 0].axis('off')
                
                # Time 2 + Prediction overlay
                axes[1, 1].imshow(np.array(img2))
                axes[1, 1].imshow(pred_mask, cmap='RdGy_r', alpha=0.6)
                axes[1, 1].set_title('Prediction')
                axes[1, 1].axis('off')
                
                # Row 3: Prediction mask and analysis
                # Binary prediction mask
                axes[2, 0].imshow(pred_mask, cmap='gray')
                axes[2, 0].set_title('binary Ground Truth')
                axes[2, 0].axis('off')
                
                # Confidence map
                colors = ['black', 'green', 'red', 'blue']
                cmap = plt.matplotlib.colors.ListedColormap(colors)
                
                # Initialize confidence map
                confidence = np.zeros_like(pred_np)
                confidence[pred_np > 0.7] = 1    # High confidence (green)
                confidence[np.logical_and(pred_np > 0.3, pred_np <= 0.7)] = 2  # Medium confidence (red)
                confidence[np.logical_and(pred_np > 0.1, pred_np <= 0.3)] = 3  # Low confidence (blue)
                
                im = axes[2, 1].imshow(confidence, cmap=cmap, vmin=0, vmax=3)
                axes[2, 1].set_title('actual predicted resulted\nGreen: High, Red: Medium, Blue: Low')
                axes[2, 1].axis('off')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=axes[2, 1], ticks=[0.5, 1.5, 2.5, 3.5])
                cbar.set_label('Confidence Levels')
                cbar.set_ticklabels(['Background', 'High', 'Medium', 'Low'])
                
                # Final progress update
                progress_bar.progress(100)
                status_text.text("Analysis complete!")
                
                # Show results
                plt.tight_layout()
                st.pyplot(fig)
                
                # Add explanation with success message
                st.success("Change detection completed successfully!")
                    
        else:
            st.info("👆 Upload both images and click 'Detect Changes' to start the analysis")

if __name__ == "__main__":
    main()