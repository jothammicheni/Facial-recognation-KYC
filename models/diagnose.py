import os
import sys

print("=== DIAGNOSTIC INFORMATION ===\n")

# 1. Check Python path
print("1. Python sys.path:")
for i, p in enumerate(sys.path[:10]):  # Show first 10 paths
    print(f"   {i}: {p}")
print("   ...\n")

# 2. Check environment variables
print("2. Relevant Environment Variables:")
env_vars = [k for k in os.environ if 'FACE' in k.upper() or 'MODEL' in k.upper() or 'PATH' in k.upper()]
for var in env_vars:
    print(f"   {var}: {os.environ.get(var)}")
print()

# 3. Check if face_recognition_models is importable
print("3. Checking face_recognition_models import:")
try:
    import face_recognition_models
    print(f"   ✓ Successfully imported")
    print(f"   Location: {face_recognition_models.__file__}")
    
    # Try to list contents
    import pkgutil
    print(f"   Package contents:")
    for importer, modname, ispkg in pkgutil.iter_modules(face_recognition_models.__path__):
        print(f"     - {modname}")
except ImportError as e:
    print(f"   ✗ Import failed: {e}")
print()

# 4. Check the models directory
print("4. Models in your custom path:")
models_path = "/home/jotham/Desktop/virtual_currency_system/facial_recog/models"
if os.path.exists(models_path):
    print(f"   Path exists: {models_path}")
    files = os.listdir(models_path)
    print(f"   Files: {files}")
else:
    print(f"   ✗ Path does not exist: {models_path}")
print()

# 5. Try to import face_recognition with debugging
print("5. Attempting to import face_recognition:")
try:
    # Monkey-patch to see what's happening
    import importlib.util
    import importlib.machinery
    
    # Try to find what face_recognition is looking for
    spec = importlib.util.find_spec("face_recognition")
    if spec:
        print(f"   Found spec: {spec.origin}")
    
    # Now try the import
    import face_recognition
    print(f"   ✓ Successfully imported face_recognition")
    print(f"   Version: {face_recognition.__version__}")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
print("\n=== END DIAGNOSTIC ===")
