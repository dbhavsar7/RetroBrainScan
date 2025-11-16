import base64
from PIL import Image
from io import BytesIO

def get_mri_brain_image_base64():
    """
    Returns a base64 encoded version of the MRI brain scan image.
    This can be used in PDF generation and API responses.
    """
    # For production, this would load from a file:
    # with open('MRI_of_Human_Brain.jpg', 'rb') as f:
    #     image_data = f.read()
    #     return base64.b64encode(image_data).decode('utf-8')
    
    # For now, return a placeholder that will be replaced with actual image
    # The image file should be placed at: Flask_Backend/static/images/MRI_of_Human_Brain.jpg
    return None

def load_demo_image(image_path='static/images/MRI_of_Human_Brain.jpg'):
    """
    Load and return the demo MRI image as bytes or base64
    """
    try:
        with open(image_path, 'rb') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Image not found at {image_path}")
        return None
