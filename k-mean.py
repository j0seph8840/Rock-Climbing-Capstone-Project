import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import glob

# Function to detect holds using k-means clustering and show masks
def detect_holds_by_kmeans(image, num_clusters=7):
    output_image = image.copy()

    # Reshape image to a 2D array of pixels
    pixels = image.reshape(-1, 3)
    
    # Apply KMeans to cluster pixels
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(pixels)
    
    # Get the cluster labels and cluster centers (colors)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_.astype('uint8')

    # Reshape labels back to the image dimensions
    clustered_image = labels.reshape(image.shape[:2])

    for cluster_idx in range(num_clusters):
        # Mask out all pixels that belong to the current cluster
        mask = (clustered_image == cluster_idx).astype(np.uint8)

        # Find non-zero indices in the mask
        y_indices, x_indices = np.where(mask > 0)

        if len(x_indices) > 0 and len(y_indices) > 0:
            # Calculate the bounding box coordinates
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)

            # Get the color for the current cluster (use as the bounding box color)
            box_color = tuple(int(c) for c in cluster_centers[cluster_idx])

            # Draw the bounding box with thicker lines
            cv2.rectangle(output_image, (x_min, y_min), (x_max, y_max), box_color, 20)

        # Show the mask for the current cluster
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(mask, cmap='gray')
        plt.title(f'Mask for Cluster {cluster_idx}')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(output_image)
        plt.title('Detected Holds with Bounding Boxes')
        plt.axis('off')

        plt.show()

    return output_image

def main():
    # Get image file paths
    image_paths = glob.glob(r'Rock-Climbing-Capstone-Project/Test Holds/*.jpg')

    for image_path in image_paths:
        # Load image
        image = cv2.imread(image_path)

        # Convert image to RGB (if needed)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect holds using K-means clustering
        output = detect_holds_by_kmeans(image_rgb, num_clusters=7)

        # Display the result
        plt.figure(figsize=(10, 5))
        plt.imshow(output)
        plt.axis('off')  # Hide axes
        plt.show()

if __name__ == "__main__":
    main()
