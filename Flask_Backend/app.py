# Flask backend
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from db_utils import DatabaseManager
import os
import base64

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'OUTPUT_IMAGES'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Initialize database manager
db = DatabaseManager('RetroBrainScanDB.db')


def allowed_file(filename):
    """Check if file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/message', methods=['POST'])
def receive_message():
    title = 'you should get this message back'
    message = request.json.get('message')
    print("Received:", message)
    return jsonify({"response": title})


@app.route('/upload', methods=['POST'])
def upload_images():
    """Handle image upload and storage"""
    try:
        # Check if files are in the request
        if 'files' not in request.files:
            return jsonify({"error": "No files provided"}), 400
        
        files = request.files.getlist('files')
        
        if not files or len(files) == 0:
            return jsonify({"error": "No files selected"}), 400
        
        uploaded_images = []
        
        for file in files:
            if file and file.filename != '':
                # Validate file
                if not allowed_file(file.filename):
                    print(f"File rejected: {file.filename} - not an allowed image format")
                    continue
                
                # Secure the filename
                filename = secure_filename(file.filename)
                
                # Read file data
                file_data = file.read()
                
                if len(file_data) == 0:
                    print(f"File rejected: {file.filename} - empty file")
                    continue
                
                # Store in database
                success = db.store_image(filename, file_data)
                
                if success:
                    # Also save to disk for backup
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    with open(filepath, 'wb') as f:
                        f.write(file_data)
                    
                    # Prepare response data (base64 encoded for display)
                    uploaded_images.append({
                        "filename": filename,
                        "data": base64.b64encode(file_data).decode('utf-8'),
                        "size": len(file_data)
                    })
                    
                    print(f"✓ Image uploaded successfully: {filename}")
                else:
                    print(f"✗ Failed to store image in database: {filename}")
        
        if not uploaded_images:
            return jsonify({"error": "No valid images were uploaded"}), 400
        
        stats = db.get_database_stats()
        
        return jsonify({
            "message": f"Successfully uploaded {len(uploaded_images)} image(s)",
            "uploaded_images": uploaded_images,
            "database_stats": stats
        }), 200
    
    except Exception as e:
        print(f"Error during upload: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.route('/images', methods=['GET'])
def get_images():
    """Get list of all uploaded images"""
    try:
        limit = request.args.get('limit', 100, type=int)
        images = db.get_all_images(limit)
        stats = db.get_database_stats()
        
        return jsonify({
            "images": images,
            "stats": stats
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/image/<int:image_id>', methods=['GET'])
def get_image(image_id):
    """Get a specific image by ID"""
    try:
        image = db.get_image(image_id)
        
        if not image:
            return jsonify({"error": "Image not found"}), 404
        
        return jsonify(image), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/stats', methods=['GET'])
def get_stats():
    """Get database statistics"""
    try:
        stats = db.get_database_stats()
        return jsonify(stats), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)

