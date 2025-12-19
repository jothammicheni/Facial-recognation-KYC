import os
import sys
import re
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify
from flask_cors import CORS

# Set the models path BEFORE importing face_recognition
os.environ['FACE_RECOGNITION_MODELS_PATH'] = '/home/jotham/Desktop/virtual_currency_system/facial_recog/models'

import face_recognition
import cv2
import numpy as np

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_FOLDER = os.path.join(BASE_DIR, 'images')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

# Create images folder if it doesn't exist
os.makedirs(IMAGES_FOLDER, exist_ok=True)

print(f"Face Recognition Models Path: {os.environ.get('FACE_RECOGNITION_MODELS_PATH')}")
print(f"Images folder: {IMAGES_FOLDER}")
print("Flask face recognition API starting...")

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def format_email_to_filename(email):
    """
    Convert email to filename format: jotham@gmail.com -> jotham_gmail_com.jpg
    """
    # Remove special characters and split
    local_part, domain = email.split('@')
    
    # Replace dots in domain with underscores
    domain_formatted = domain.replace('.', '_')
    
    # Create filename
    filename = f"{local_part}_{domain_formatted}.jpg"
    return filename

def extract_email_from_filename(filename):
    """
    Convert filename back to email: jotham_gmail_com.jpg -> jotham@gmail.com
    """
    try:
        # Remove extension
        name_without_ext = os.path.splitext(filename)[0]
        
        # Split by underscore
        parts = name_without_ext.split('_')
        
        if len(parts) >= 2:
            local_part = parts[0]
            domain_parts = parts[1:]
            domain = '.'.join(domain_parts)
            return f"{local_part}@{domain}"
    except:
        pass
    return None

def load_known_faces():
    """
    Load all face encodings from images in the images folder
    Returns: (face_encodings_list, face_names_list)
    """
    known_face_encodings = []
    known_face_names = []  # Will store emails
    
    print(f"Loading known faces from {IMAGES_FOLDER}...")
    
    for filename in os.listdir(IMAGES_FOLDER):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            image_path = os.path.join(IMAGES_FOLDER, filename)
            
            try:
                # Load image
                image = face_recognition.load_image_file(image_path)
                
                # Get face encodings
                encodings = face_recognition.face_encodings(image)
                
                if len(encodings) > 0:
                    # Use the first face found
                    known_face_encodings.append(encodings[0])
                    
                    # Convert filename back to email for identification
                    email = extract_email_from_filename(filename)
                    if email:
                        known_face_names.append(email)
                        print(f"  ✓ Loaded: {email} from {filename}")
                    else:
                        known_face_names.append(filename)
                        print(f"  ✓ Loaded: {filename}")
                else:
                    print(f"  ✗ No faces found in: {filename}")
                    
            except Exception as e:
                print(f"  ✗ Error loading {filename}: {e}")
    
    print(f"Total known faces loaded: {len(known_face_encodings)}")
    return known_face_encodings, known_face_names

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'images_folder': IMAGES_FOLDER,
        'total_images': len([f for f in os.listdir(IMAGES_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))])
    })

@app.route('/api/selfie/upload', methods=['POST'])
def register_user():
    """
    Register a new user with email and selfie
    Request: multipart/form-data with 'email' and 'image' fields
    """
    try:
        # Check if request has files
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            }), 400
        
        # Get email from form data
        email = request.form.get('email')
        if not email or '@' not in email:
            return jsonify({
                'success': False,
                'error': 'Valid email is required'
            }), 400
        
        # Clean and validate email
        email = email.strip().lower()
        
        # Check if user already exists
        existing_filename = format_email_to_filename(email)
        existing_path = os.path.join(IMAGES_FOLDER, existing_filename)
        if os.path.exists(existing_path):
            return jsonify({
                'success': False,
                'error': 'User with this email already registered'
            }), 400
        
        # Get the image file
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No image selected'
            }), 400
        
        if not allowed_file(image_file.filename):
            return jsonify({
                'success': False,
                'error': f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Verify the image contains a face
        try:
            # Read image file
            file_bytes = np.frombuffer(image_file.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if image is None:
                return jsonify({
                    'success': False,
                    'error': 'Invalid image file'
                }), 400
            
            # Convert to RGB for face_recognition
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Check for faces
            face_locations = face_recognition.face_locations(rgb_image)
            
            if len(face_locations) == 0:
                return jsonify({
                    'success': False,
                    'error': 'No face detected in the image. Please provide a clear selfie.'
                }), 400
            
            if len(face_locations) > 1:
                return jsonify({
                    'success': False,
                    'error': 'Multiple faces detected. Please provide a selfie with only one person.'
                }), 400
            
            # Get face encoding to verify it's valid
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            if len(face_encodings) == 0:
                return jsonify({
                    'success': False,
                    'error': 'Could not extract face features. Please try with a clearer image.'
                }), 400
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Error processing image: {str(e)}'
            }), 400
        
        # Reset file pointer
        image_file.seek(0)
        
        # Format filename
        filename = format_email_to_filename(email)
        
        # Save the image
        filepath = os.path.join(IMAGES_FOLDER, filename)
        image_file.save(filepath)
        
        print(f"✓ New user registered: {email} -> {filename}")
        
        return jsonify({
            'success': True,
            'message': 'Registration successful',
            'email': email,
            'filename': filename,
            'filepath': filepath
        }), 201
        
    except Exception as e:
        print(f"Registration error: {e}")
        return jsonify({
            'success': False,
            'error': f'Registration failed: {str(e)}'
        }), 500

@app.route('/api/selfie/login', methods=['POST'])
def login_user():
    """
    Login user by matching face with registered images
    Request: multipart/form-data with 'image' field
    """
    try:
        # Check if request has files
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            }), 400
        
        # Get the image file
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No image selected'
            }), 400
        
        if not allowed_file(image_file.filename):
            return jsonify({
                'success': False,
                'error': f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Load known faces
        known_face_encodings, known_face_names = load_known_faces()
        
        if not known_face_encodings:
            return jsonify({
                'success': False,
                'error': 'No registered users found'
            }), 400
        
        # Process the uploaded image
        try:
            # Read image file
            file_bytes = np.frombuffer(image_file.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if image is None:
                return jsonify({
                    'success': False,
                    'error': 'Invalid image file'
                }), 400
            
            # Convert to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Find faces in the image
            face_locations = face_recognition.face_locations(rgb_image)
            
            if len(face_locations) == 0:
                return jsonify({
                    'success': False,
                    'error': 'No face detected in the image'
                }), 400
            
            if len(face_locations) > 1:
                return jsonify({
                    'success': False,
                    'error': 'Multiple faces detected. Please provide an image with only one person.'
                }), 400
            
            # Get face encoding
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            if len(face_encodings) == 0:
                return jsonify({
                    'success': False,
                    'error': 'Could not extract face features'
                }), 400
            
            # Compare with known faces
            face_encoding = face_encodings[0]
            
            # Method 1: Compare faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            
            # Method 2: Calculate face distances for better accuracy
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            
            # Find the best match
            best_match_index = np.argmin(face_distances)
            best_distance = face_distances[best_match_index]
            
            # Threshold for matching (lower is better, typically < 0.6 is a good match)
            MATCH_THRESHOLD = 0.6
            
            if matches[best_match_index] and best_distance < MATCH_THRESHOLD:
                matched_email = known_face_names[best_match_index]
                confidence = (1 - best_distance) * 100
                
                print(f"✓ Login successful: {matched_email} (confidence: {confidence:.1f}%)")
                
                return jsonify({
                    'success': True,
                    'authenticated': True,
                    'email': matched_email,
                    'confidence': round(confidence, 1),
                    'message': 'Authentication successful'
                })
            else:
                print(f"✗ Login failed: No match found (best distance: {best_distance:.4f})")
                return jsonify({
                    'success': True,
                    'authenticated': False,
                    'message': 'No matching face found'
                })
                
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Error processing image: {str(e)}'
            }), 400
        
    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({
            'success': False,
            'error': f'Login failed: {str(e)}'
        }), 500

@app.route('/api/users', methods=['GET'])
def list_users():
    """List all registered users"""
    try:
        users = []
        
        for filename in os.listdir(IMAGES_FOLDER):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                email = extract_email_from_filename(filename)
                if email:
                    filepath = os.path.join(IMAGES_FOLDER, filename)
                    file_size = os.path.getsize(filepath)
                    
                    users.append({
                        'email': email,
                        'filename': filename,
                        'file_size': file_size,
                        'registered_date': os.path.getctime(filepath)
                    })
        
        return jsonify({
            'success': True,
            'users': users,
            'total_users': len(users)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/user/<email>', methods=['DELETE'])
def delete_user(email):
    """Delete a registered user by email"""
    try:
        filename = format_email_to_filename(email)
        filepath = os.path.join(IMAGES_FOLDER, filename)
        
        if not os.path.exists(filepath):
            return jsonify({
                'success': False,
                'error': 'User not found'
            }), 404
        
        os.remove(filepath)
        
        print(f"✓ User deleted: {email}")
        
        return jsonify({
            'success': True,
            'message': f'User {email} deleted successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def create_test_images():
    """Create test images for demonstration"""
    test_images = [
        ('john_doe@gmail.com', 'test_images/john.jpg'),
        ('jane_smith@gmail.com', 'test_images/jane.jpg'),
        ('jotham@gmail.com', 'test_images/jotham.jpg')
    ]
    
    for email, test_image_path in test_images:
        if os.path.exists(test_image_path):
            filename = format_email_to_filename(email)
            dest_path = os.path.join(IMAGES_FOLDER, filename)
            
            # Copy test image to images folder
            import shutil
            shutil.copy2(test_image_path, dest_path)
            print(f"Created test image: {email} -> {filename}")

if __name__ == '__main__':
    # Create test images if needed
    # create_test_images()
    
    # Load known faces on startup
    known_encodings, known_names = load_known_faces()
    
    print(f"\n=== Flask Face Recognition API ===")
    print(f"Images folder: {IMAGES_FOLDER}")
    print(f"Loaded {len(known_encodings)} registered faces")
    print(f"API endpoints:")
    print(f"  POST /api/register   - Register new user with email and selfie")
    print(f"  POST /api/login      - Login with selfie")
    print(f"  GET  /api/users      - List all registered users")
    print(f"  DELETE /api/user/<email> - Delete user")
    print(f"  GET  /api/health     - Health check")
    print(f"\nStarting server on http://localhost:5000")
    print("Press Ctrl+C to stop\n")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5006, debug=True)