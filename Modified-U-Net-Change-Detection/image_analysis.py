import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from scipy import ndimage
import torch
import torch.nn.functional as F

class AdvancedImageAnalysis:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def extract_texture_features(self, image):
        """Extract GLCM texture features"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        glcm = graycomatrix(gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])
        
        return {
            'contrast': graycoprops(glcm, 'contrast').mean(),
            'dissimilarity': graycoprops(glcm, 'dissimilarity').mean(),
            'homogeneity': graycoprops(glcm, 'homogeneity').mean(),
            'correlation': graycoprops(glcm, 'correlation').mean(),
            'energy': graycoprops(glcm, 'energy').mean()
        }

    def analyze_change_patterns(self, change_mask):
        """Analyze patterns in detected changes"""
        labeled_mask, num_features = ndimage.label(change_mask)
        
        # Calculate properties for each changed region
        regions = []
        for i in range(1, num_features + 1):
            region = (labeled_mask == i)
            area = np.sum(region)
            centroid = ndimage.center_of_mass(region)
            
            regions.append({
                'area': area,
                'centroid': centroid,
                'compactness': self._calculate_compactness(region)
            })
            
        return regions

    def _calculate_compactness(self, region):
        """Calculate shape compactness"""
        perimeter = np.sum(cv2.findContours(region.astype(np.uint8), 
                                          cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_SIMPLE)[0])
        area = np.sum(region)
        return 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0