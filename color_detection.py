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
    result = cv2.addWeighted(masked_image, 0.7, gray_image, 0.3, 0)
    
    return result

def main():
    directory = "/Users/josephpalacios/Desktop/School/Fall 2024/Rock-Climbing-Capstone-Project/Holds"
    image_files = glob.glob(f"{directory}/*.jpg") + glob.glob(f"{directory}/*.JPG")

    if not image_files:
        print(f"No JPG files found in {directory}")
        return

    for file in image_files:
        image = cv2.imread(file)
        if image is None:
            print(f"Failed to load image: {file}")
            continue

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        colors, labels = cluster_colors(rgb_image, n_clusters=5)  # Adjust n_clusters as needed
        
        result_image = create_color_masked_image(rgb_image, colors, labels)
        
        # Display the result
        plt.figure(figsize=(12, 6))
        plt.subplot(121), plt.imshow(rgb_image), plt.title('Original Image')
        plt.subplot(122), plt.imshow(result_image), plt.title('Color Masked Image')
        plt.show()

if __name__ == "__main__":
    main()
