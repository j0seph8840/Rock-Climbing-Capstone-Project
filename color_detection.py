import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

# Function to detect holds and draw bounding boxes based on color
def detect_holds_by_color(image, color_ranges):
    output_image = image.copy()

    for color_name, (lower, upper) in color_ranges.items():
        # Create mask for the specific color
        lower_bound = np.array(lower, dtype="uint8")
        upper_bound = np.array(upper, dtype="uint8")
        mask = cv2.inRange(image, lower_bound, upper_bound)

        # Find where the mask has non-zero values (i.e., the areas with the specified color)
        y_indices, x_indices = np.where(mask > 0)

        if len(x_indices) > 0 and len(y_indices) > 0:
            # Calculate the bounding box coordinates based on min/max of x and y
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)

            # Convert the HSV color to RGB for bounding box color
            color_rgb = cv2.cvtColor(np.uint8([[lower_bound]]), cv2.COLOR_HSV2RGB)[0][0]
            color_rgb = tuple(int(c) for c in color_rgb)  # Convert to tuple for OpenCV

            # Draw the bounding box with thicker lines (e.g., thickness=5) and respective color
            cv2.rectangle(output_image, (x_min, y_min), (x_max, y_max), color_rgb, 20)

    return output_image, mask

def main():
    # Define color ranges for HSV (Adjust these values as needed)
    color_ranges = {
        "red": ([0, 100, 100], [10, 255, 255]),         # Example for red
        "green": ([35, 100, 100], [85, 255, 255]),      # Example for green
        "purple": ([130, 100, 100], [160, 255, 255]),   # Example for purple
        "yellow": ([20, 100, 100], [30, 255, 255]),     # Example for yellow
        "blue": ([100, 100, 100], [130, 255, 255]),     # Example for blue
        "orange": ([10, 150, 150], [25, 255, 255]),     # Example for orange
        "black": ([0, 0, 0], [180, 255, 50]),           # Example for black
        "white": ([0, 0, 200], [180, 20, 255])          # Example for white
    }

    # Get image file paths
    images = glob.glob('Rock-Climbing-Capstone-Project/Holds/*.jpg')

    for image_path in images:
        # Load image
        bgr_image = cv2.imread(image_path)

        # Convert image to HSV
        image_hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

        # Detect holds by color and get the mask and output image
        output, mask = detect_holds_by_color(image_hsv, color_ranges)

        # Convert mask to displayable format (binary mask to grayscale)
        mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        # Display the original image, the mask, and the bounding box result
        plt.figure(figsize=(15, 5))

        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')

        # Mask of the color detection
        plt.subplot(1, 3, 2)
        plt.imshow(mask_display)
        plt.title('Color Mask')
        plt.axis('off')

        # Final image with bounding box
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(output, cv2.COLOR_HSV2RGB))
        plt.title('Bounding Box on Original Image')
        plt.axis('off')

        plt.show()

if __name__ == "__main__":
    main()
