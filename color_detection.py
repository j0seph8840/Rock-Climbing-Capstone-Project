import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import glob

def cluster_colors(image, n_clusters=5):
    # Reshape the image to be a list of pixels
    pixels = image.reshape((-1, 3))
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(pixels)
    
    # Get the colors
    colors = kmeans.cluster_centers_
    
    # Convert to integer RGB values
    colors = colors.round().astype(int)
    
    return colors, labels

def create_color_masked_image(image, colors, labels):
    # Create a copy of the image
    masked_image = image.copy()
    
    # Reshape labels to match image shape
    labels = labels.reshape(image.shape[:2])
    
    # Create a grayscale version of the image
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
    
    # For each color, create a mask and apply it
    for i, color in enumerate(colors):
        mask = (labels == i).astype(np.uint8) * 255
        color_mask = np.dstack([mask] * 3)
        masked_image = np.where(color_mask == 255, image, masked_image)
    
    # Blend the masked image with the grayscale image
    resu