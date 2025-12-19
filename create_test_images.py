import cv2
import numpy as np

# Create a simple colored square as a test image
def create_test_image(filename, color=(255, 0, 0)):
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[50:150, 50:150] = color
    cv2.imwrite(filename, img)
    print(f"Created: {filename}")

# Create test images
create_test_image("person1.jpg", (255, 0, 0))  # Blue square
create_test_image("person2.jpg", (0, 255, 0))  # Green square
create_test_image("unknown_photo.jpg", (0, 0, 255))  # Red square

print("Test images created!")
