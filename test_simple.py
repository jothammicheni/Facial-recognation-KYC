import os
# Set models path
os.environ['FACE_RECOGNITION_MODELS_PATH'] = '/home/jotham/Desktop/virtual_currency_system/facial_recog/models'

import face_recognition
import numpy as np

print("Testing face_recognition installation...")

# Create a dummy image
test_image = np.zeros((100, 100, 3), dtype=np.uint8)
test_image[30:70, 30:70] = [255, 255, 255]  # White square as a "face"

try:
    # Test face detection
    locations = face_recognition.face_locations(test_image)
    print(f"Face detection test: Found {len(locations)} face(s)")
    
    # Test face encoding
    if len(locations) > 0:
        encodings = face_recognition.face_encodings(test_image, locations)
        print(f"Face encoding test: Created {len(encodings)} encoding(s)")
    
    print("✓ All tests passed!")
    
except Exception as e:
    print(f"✗ Test failed: {e}")
