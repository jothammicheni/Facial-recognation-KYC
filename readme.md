# 1. Go back to your project directory
cd ~/Desktop/virtual_currency_system/facial_recog

# 2. Remove the incorrect path from sys.path
# (You added models directory to Python path, which is incorrect)

# 3. Check what's actually in your site-packages
ls -la venv/lib/python3.12/site-packages/ | grep -i face

# 4. Completely reinstall face_recognition_models PROPERLY
pip uninstall face_recognition_models face_recognition -y

# 5. Install setuptools first (missing pkg_resources)
pip install setuptools

# 6. Install face_recognition_models from GitHub WITH build dependencies
pip install --no-binary :all: git+https://github.com/ageitgey/face_recognition_models

# 7. Verify the installation
python3 -c "import face_recognition_models; print('SUCCESS: Imported from', face_recognition_models.__file__)"

# 8. Check if models are in the package
python3 -c "
import face_recognition_models
import os
pkg_path = os.path.dirname(face_recognition_models.__file__)
models_dir = os.path.join(pkg_path, 'models')
print('Package path:', pkg_path)
print('Models directory exists:', os.path.exists(models_dir))
if os.path.exists(models_dir):
    print('Model files:', os.listdir(models_dir))
"

# 9. Install face_recognition
pip install face_recognition

# 10. Test the full import
python3 -c "
import face_recognition
print('✓ face_recognition imported successfully')
print('Version:', face_recognition.__version__)
"