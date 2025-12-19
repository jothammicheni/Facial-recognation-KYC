import os
import sys
import time

# Set the models path BEFORE importing face_recognition
os.environ['FACE_RECOGNITION_MODELS_PATH'] = '/home/jotham/Desktop/virtual_currency_system/facial_recog/models'

import face_recognition
import cv2
import numpy as np

print("Face Recognition Models Path:", os.environ.get('FACE_RECOGNITION_MODELS_PATH'))
print("Starting face recognition with webcam...")

# Check if image files exist
def check_file_exists(filename):
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found!")
        print(f"Current directory: {os.getcwd()}")
        print(f"Files in directory: {os.listdir('.')}")
        return False
    return True

# Define your image file names
known_images = ["person1.jpg", "person2.jpg", "person3.jpg"]

# Check if known images exist
missing_files = []
for img in known_images:
    if not check_file_exists(img):
        missing_files.append(img)

if missing_files:
    print(f"\nMissing files: {missing_files}")
    print("Please make sure these image files exist in the same directory as recog.py")
    sys.exit(1)

def load_known_faces():
    """Load known faces from image files"""
    known_face_encodings = []
    known_face_names = []
    
    print("Loading known faces...")
    
    for i, img_file in enumerate(known_images):
        print(f"  Loading {img_file}...")
        known_image = face_recognition.load_image_file(img_file)
        
        # Get face encodings
        encodings = face_recognition.face_encodings(known_image)
        
        if len(encodings) == 0:
            print(f"  Warning: No faces found in {img_file}")
        else:
            known_face_encodings.append(encodings[0])
            known_face_names.append(f"Person {i+1}")
            print(f"  ✓ Face encoding extracted from {img_file}")
    
    if not known_face_encodings:
        print("Error: No faces found in any known images!")
        return None, None
    
    print(f"Loaded {len(known_face_encodings)} known faces")
    return known_face_encodings, known_face_names

def capture_and_match(known_face_encodings, known_face_names):
    """Capture a selfie from webcam and match against known faces"""
    # Initialize webcam
    video_capture = cv2.VideoCapture(0)
    
    if not video_capture.isOpened():
        print("Error: Could not open webcam!")
        return False
    
    print("\nWebcam opened successfully!")
    print("Press 'c' to capture a selfie")
    print("Press 'q' to quit")
    
    frame_count = 0
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()
        
        if not ret:
            print("Failed to grab frame")
            break
        
        # Resize frame for faster processing (optional)
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        
        # Convert BGR to RGB (face_recognition uses RGB)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find all face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        # Process each face in the frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Scale back up face locations since we scaled down the frame
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2
            
            # Compare the face with known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            
            # Use face distance for better matching
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index] and face_distances[best_match_index] < 0.6:
                    name = known_face_names[best_match_index]
            
            # Draw rectangle and name
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw label background
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
        
        # Display instructions on frame
        cv2.putText(frame, "Press 'c' to capture selfie", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display the resulting image
        cv2.imshow('Face Recognition - Webcam', frame)
        
        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c'):
            # Capture selfie
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            selfie_filename = f"selfie_{timestamp}.jpg"
            cv2.imwrite(selfie_filename, frame)
            print(f"\n✓ Selfie captured: {selfie_filename}")
            
            # Process the captured selfie
            process_captured_selfie(selfie_filename, known_face_encodings, known_face_names)
            
            # Show confirmation
            cv2.putText(frame, "SELFIE CAPTURED!", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow('Face Recognition - Webcam', frame)
            cv2.waitKey(1000)  # Show message for 1 second
            
        elif key == ord('q'):
            print("\nQuitting webcam...")
            break
        
        frame_count += 1
    
    # Release webcam and close windows
    video_capture.release()
    cv2.destroyAllWindows()
    return True

def process_captured_selfie(selfie_filename, known_face_encodings, known_face_names):
    """Process the captured selfie and match against known faces"""
    print(f"\nProcessing captured selfie: {selfie_filename}")
    
    # Load the captured image
    selfie_image = face_recognition.load_image_file(selfie_filename)
    
    # Find faces in the selfie
    face_locations = face_recognition.face_locations(selfie_image)
    face_encodings = face_recognition.face_encodings(selfie_image, face_locations)
    
    if len(face_encodings) == 0:
        print("No faces found in the captured selfie!")
        return
    
    print(f"Found {len(face_encodings)} face(s) in selfie")
    
    # Process each face found
    for i, (face_location, face_encoding) in enumerate(zip(face_locations, face_encodings)):
        top, right, bottom, left = face_location
        
        # Compare with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index] and face_distances[best_match_index] < 0.6:
                name = known_face_names[best_match_index]
        
        # Calculate match confidence
        if name != "Unknown":
            confidence = (1 - face_distances[best_match_index]) * 100
            print(f"  Face {i+1}: Matched as '{name}' with {confidence:.1f}% confidence")
        else:
            print(f"  Face {i+1}: Unknown person (no match found)")
        
        # Draw on the selfie image for display
        selfie_display = cv2.cvtColor(selfie_image, cv2.COLOR_RGB2BGR)
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(selfie_display, (left, top), (right, bottom), color, 2)
        
        # Label with name and confidence
        label = f"{name}"
        if name != "Unknown":
            label += f" ({confidence:.1f}%)"
        
        cv2.rectangle(selfie_display, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(selfie_display, label, (left + 6, bottom - 6), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
    
    # Save annotated selfie
    annotated_filename = f"annotated_{selfie_filename}"
    cv2.imwrite(annotated_filename, selfie_display)
    print(f"✓ Annotated selfie saved: {annotated_filename}")
    
    # Display the annotated selfie
    cv2.imshow('Captured Selfie - Results', selfie_display)
    print("Press any key in the results window to continue...")
    cv2.waitKey(0)
    cv2.destroyWindow('Captured Selfie - Results')

def real_time_recognition(known_face_encodings, known_face_names):
    """Continuous real-time face recognition from webcam"""
    video_capture = cv2.VideoCapture(0)
    
    if not video_capture.isOpened():
        print("Error: Could not open webcam!")
        return
    
    print("\n=== REAL-TIME FACE RECOGNITION MODE ===")
    print("Press 'r' to toggle real-time recognition")
    print("Press 'c' to capture selfie")
    print("Press 'q' to quit")
    
    process_this_frame = True
    
    while True:
        ret, frame = video_capture.read()
        
        if not ret:
            break
        
        # Only process every other frame to save processing power
        if process_this_frame:
            # Resize for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Find faces
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index] and face_distances[best_match_index] < 0.6:
                        name = known_face_names[best_match_index]
                
                face_names.append(name)
        
        process_this_frame = not process_this_frame
        
        # Display results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
        
        # Display FPS
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Display mode
        cv2.putText(frame, "Mode: Real-time Recognition", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'c' to capture | 'q' to quit", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Real-time Face Recognition', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Capture selfie
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            selfie_filename = f"realtime_selfie_{timestamp}.jpg"
            cv2.imwrite(selfie_filename, frame)
            print(f"\n✓ Selfie captured during real-time recognition: {selfie_filename}")
    
    video_capture.release()
    cv2.destroyAllWindows()

def main():
    """Main function"""
    try:
        # Load known faces
        known_face_encodings, known_face_names = load_known_faces()
        
        if known_face_encodings is None:
            return
        
        print("\n=== FACE RECOGNITION SYSTEM ===")
        print("1. Capture Selfie and Match")
        print("2. Real-time Recognition")
        print("3. Exit")
        
        while True:
            choice = input("\nSelect mode (1/2/3): ").strip()
            
            if choice == '1':
                # Mode 1: Capture selfie and match
                capture_and_match(known_face_encodings, known_face_names)
                
            elif choice == '2':
                # Mode 2: Real-time recognition
                real_time_recognition(known_face_encodings, known_face_names)
                
            elif choice == '3':
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()