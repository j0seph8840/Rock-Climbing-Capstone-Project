import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

# Function to detect holds and draw bounding boxes based on color
def detect_holds_by_color(image, color_ranges, min_contour_area=20000):
    output_image = image.copy()

    # Extract only the Hue channel
    hue_channel = image[:, :, 0]

    # Structuring element for morphology
    kernel = np.ones((5, 5), np.uint8)

    for color_name, ranges in color_ranges.items():
        if color_name in ['black', 'white']:
            # Use all channels since we can differientate with just Hue
            mask = cv2.inRange(image, np.array(ranges[0]), np.array(ranges[1]))
            lower = ranges[0]
        elif color_name == 'red':
            # Handle red's wrap-around
            lower1, upper1, lower2, upper2 = ranges
            mask1 = cv2.inRange(hue_channel, lower1[0], upper1[0])
            mask2 = cv2.inRange(hue_channel, lower2[0], upper2[0])
            mask = cv2.bitwise_or(mask1, mask2)
            lower = lower1
        else:
            # For other colors, use only Hue
            lower, upper = ranges
            mask = cv2.inRange(hue_channel, lower[0], upper[0]) 

        # DEBUG: Display each mask before morphology to ensure thresholding is correct
        plt.imshow(mask, cmap='gray')
        plt.title(f'Mask for {color_name}')
        plt.axis('off')
        plt.show()

        # Apply morphological operations to clean up the mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Removes noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # Closes gaps

        # Find contours in the cleaned-up mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Loop through contours and draw bounding boxes
        for contour in contours:
            if cv2.contourArea(contour) > min_contour_area:  # Filter out small areas
                x, y, w, h = cv2.boundingRect(contour)

                # Convert HSV to RGB for bounding box color
                color_rgb = cv2.cvtColor(np.uint8([[lower]]), cv2.COLOR_HSV2RGB)[0][0]
                color_rgb = tuple(int(c) for c in color_rgb)  # Convert to tuple for OpenCV

                # Draw the bounding box with thicker lines
                cv2.rectangle(output_image, (x, y), (x + w, y + h), color_rgb, 15)

    return output_image

def main():
    # Define color ranges for HSV (Adjust these values as needed)
    color_ranges = {
        "red": ([0, 0, 0], [10, 255, 255], [160, 0, 0], [180, 255, 255]),   # Red has a wrap-around
        "orange": ([10, 0, 0], [25, 255, 255]),
        "yellow": ([20, 0, 0], [30, 255, 255]), 
        "green": ([35, 0, 0], [85, 255, 255]),    
        "cyan": ([85, 0, 0], [100, 255, 255]),
        "blue": ([100, 0, 0], [130, 255, 255]), 
        "purple": ([130, 0, 0], [160, 255, 255]),
        "pink": ([160, 0, 0], [180, 255, 255]),  
        "black": ([0, 0, 0], [180, 255, 50]),         
        "white": ([0, 0, 200], [180, 20, 255])        
    }

    # Get image file paths
    images = glob.glob('Rock-Climbing-Capstone-Project/Test Holds/*.jpg')

    for image_path in images:
        # Load image
        bgr_image = cv2.imread(image_path)

        # Convert image to HSV
        image_hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

        # Detect holds by color and get the output image with bounding boxes
        output = detect_holds_by_color(image_hsv, color_ranges)

        # Display the original image and final image with bounding boxes
        plt.figure(figsize=(10, 5))

        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')

        # Final image with bounding box
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(output, cv2.COLOR_HSV2RGB))
        plt.title('Bounding Boxes on Detected Holds')
        plt.axis('off')

        plt.show()

if __name__ == "__main__":
    main()
