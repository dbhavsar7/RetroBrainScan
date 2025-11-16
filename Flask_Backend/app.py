# Flask backend
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from db_utils import DatabaseManager
import os
import base64
import json
from io import BytesIO
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib import colors
from datetime import datetime

# Google Gemini AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-generativeai not installed. Install with: pip install google-generativeai")

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

# Google Gemini API Configuration
GOOGLE_AI_API_KEY = "AIzaSyBO6oLZrlSuPN1jPZK59NFxVQtwpQvKYaI"
# Using Gemini 2.5 Pro - try gemini-2.0-flash-exp or gemini-1.5-pro-latest if this doesn't work
GEMINI_MODEL = "gemini-2.0-flash-exp"  # Try "gemini-1.5-pro-latest" or "gemini-1.5-pro" if unavailable

if GEMINI_AVAILABLE:
    genai.configure(api_key=GOOGLE_AI_API_KEY)


def allowed_file(filename):
    """Check if file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_clinical_prompt():
    """Load the clinical prompt template"""
    prompt_path = os.path.join(os.path.dirname(__file__), 'prompts', 'clinical_prompts.txt')
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: Clinical prompt file not found at {prompt_path}")
        return ""


def generate_care_plan(analysis_results):
    """
    Generate a clinical care plan using Google Gemini API
    
    Args:
        analysis_results: dict containing:
            - current_risk_score: float
            - current_prediction: str
            - future_risk_score: float
            - future_prediction: str
            - current_regions: list
            - future_regions: list
    
    Returns:
        str: Generated clinical care plan
    """
    if not GEMINI_AVAILABLE:
        return "Error: Google Gemini API is not available. Please install google-generativeai package."
    
    try:
        # Load the clinical prompt template
        system_prompt = load_clinical_prompt()
        
        # Format the data for the prompt
        current_regions = analysis_results.get("current_regions", [])
        future_regions = analysis_results.get("future_regions", [])
        
        # Structure regions data as specified in prompt
        regions_data = {
            "current": {
                "most_likely_regions": current_regions[:2] if len(current_regions) >= 2 else current_regions,
                "secondary_regions": current_regions[2:] if len(current_regions) > 2 else []
            },
            "future": {
                "most_likely_regions": future_regions[:2] if len(future_regions) >= 2 else future_regions,
                "secondary_regions": future_regions[2:] if len(future_regions) > 2 else []
            }
        }
        
        # Combine system prompt and user message
        full_prompt = f"""{system_prompt}

---

Please generate a clinical care plan based on the following analysis results:

1. CLINICAL RISK METRICS:
   - Current Risk Score: {analysis_results.get('current_risk_score', 0):.3f}
   - Future Risk Score: {analysis_results.get('future_risk_score', 0):.3f}
   - Current Classification: {analysis_results.get('current_prediction', 'N/A')}
   - Future Classification: {analysis_results.get('future_prediction', 'N/A')}

2. BRAIN REGIONS DATA (JSON):
{json.dumps(regions_data, indent=2)}

Please provide a comprehensive clinical report following the format specified above."""
        
        # Initialize the model
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        # Generate the care plan
        response = model.generate_content(full_prompt)
        
        return response.text
        
    except Exception as e:
        error_msg = f"Error generating care plan: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return f"Error: {error_msg}"


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


@app.route('/generate-report', methods=['POST'])
def generate_report():
    """Generate a PDF report from JSON data"""
    try:
        data = request.json
        
        # Extract data from request
        patient = data.get('patient', {})
        scan_info = data.get('scanInfo', {})
        analysis = data.get('analysis', {})
        recommendations = data.get('recommendations', [])
        radiologist = data.get('radiologist', {})
        analysis_results = data.get('analysisResults', {})
        care_plan = data.get('care_plan', '')
        
        # Create a BytesIO buffer to store the PDF
        pdf_buffer = BytesIO()
        
        # Create PDF document
        doc = SimpleDocTemplate(
            pdf_buffer,
            pagesize=letter,
            rightMargin=0.5*inch,
            leftMargin=0.5*inch,
            topMargin=0.5*inch,
            bottomMargin=0.5*inch
        )
        
        # Container for PDF content
        elements = []
        
        # Define styles
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#0B3555'),
            spaceAfter=10,
            alignment=1,  # Center
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#0B3555'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold',
            borderPadding=5,
            borderColor=colors.HexColor('#0B3555'),
            borderWidth=1
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#333333'),
            spaceAfter=6
        )
        
        # Title
        elements.append(Paragraph("🧠 BRAIN SCAN ANALYSIS REPORT", title_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Helper function to add base64 images
        def add_base64_image(base64_str, width=2.5*inch, height=2.5*inch):
            if base64_str:
                try:
                    img_data = base64.b64decode(base64_str)
                    img_buffer = BytesIO(img_data)
                    img = Image(img_buffer, width=width, height=height)
                    return img
                except Exception as e:
                    print(f"Warning: Could not add base64 image: {e}")
            return None
        
        # Brain Scan Images Section
        if analysis_results:
            elements.append(Paragraph("BRAIN SCAN IMAGES", heading_style))
            images_row = []
            
            # Current MRI
            if analysis_results.get('original_image'):
                img = add_base64_image(analysis_results['original_image'])
                if img:
                    images_row.append(img)
            
            # Current Heatmap
            if analysis_results.get('current_heatmap'):
                img = add_base64_image(analysis_results['current_heatmap'])
                if img:
                    images_row.append(img)
            
            if images_row:
                # Create a table with 2 columns for images
                img_table = Table([images_row], colWidths=[2.5*inch, 2.5*inch])
                img_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ]))
                elements.append(img_table)
                elements.append(Spacer(1, 0.1*inch))
            
            # Future MRI and Future Heatmap
            future_images_row = []
            if analysis_results.get('future_mri'):
                img = add_base64_image(analysis_results['future_mri'])
                if img:
                    future_images_row.append(img)
            
            if analysis_results.get('future_heatmap'):
                img = add_base64_image(analysis_results['future_heatmap'])
                if img:
                    future_images_row.append(img)
            
            if future_images_row:
                future_img_table = Table([future_images_row], colWidths=[2.5*inch, 2.5*inch])
                future_img_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ]))
                elements.append(future_img_table)
            
            elements.append(Spacer(1, 0.2*inch))
        else:
            # Fallback to default image if no analysis results
            try:
                image_path = os.path.join(os.path.dirname(__file__), 'static', 'images', 'MRI_of_Human_Brain.jpg')
                if os.path.exists(image_path):
                    img = Image(image_path, width=3.5*inch, height=3.5*inch)
                    elements.append(img)
                    elements.append(Spacer(1, 0.2*inch))
            except Exception as e:
                print(f"Warning: Could not load brain scan image: {e}")
        
        # Report Info
        report_info = [
            ['Report Date:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['Report ID:', patient.get('mrn', 'N/A')],
            ['Status:', 'FINAL']
        ]
        report_table = Table(report_info, colWidths=[2*inch, 4*inch])
        report_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        elements.append(report_table)
        elements.append(Spacer(1, 0.2*inch))
        
        # Patient Information Section
        elements.append(Paragraph("PATIENT INFORMATION", heading_style))
        patient_info_table = [
            ['Name:', patient.get('name', 'N/A')],
            ['Age:', str(patient.get('age', 'N/A'))],
            ['Gender:', patient.get('gender', 'N/A')],
            ['MRN:', patient.get('mrn', 'N/A')],
            ['DOB:', patient.get('dateOfBirth', 'N/A')],
            ['Hospital:', patient.get('hospital', 'N/A')]
        ]
        patient_table = Table(patient_info_table, colWidths=[1.5*inch, 4.5*inch])
        patient_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#E6F0F7')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey)
        ]))
        elements.append(patient_table)
        elements.append(Spacer(1, 0.15*inch))
        
        # Scan Information Section
        elements.append(Paragraph("SCAN INFORMATION", heading_style))
        scan_data = [
            ['Scan Type:', scan_info.get('scanType', 'N/A')],
            ['Date:', scan_info.get('scanDate', 'N/A')],
            ['Time:', scan_info.get('scanTime', 'N/A')],
            ['Duration:', scan_info.get('scanDuration', 'N/A')],
            ['Scanner:', scan_info.get('scannerModel', 'N/A')]
        ]
        scan_table = Table(scan_data, colWidths=[1.5*inch, 4.5*inch])
        scan_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#E6F0F7')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey)
        ]))
        elements.append(scan_table)
        elements.append(Spacer(1, 0.15*inch))
        
        # Findings Section
        elements.append(Paragraph("ANALYSIS FINDINGS", heading_style))
        findings = analysis.get('findings', [])
        for finding in findings:
            finding_text = f"<b>{finding.get('title', 'N/A')} - {finding.get('status', 'N/A')}</b><br/>"
            finding_text += finding.get('description', 'N/A')
            finding_text += f"<br/><i>Confidence: {finding.get('confidence', 0)}%</i>"
            elements.append(Paragraph(finding_text, normal_style))
            elements.append(Spacer(1, 0.1*inch))
        
        elements.append(Spacer(1, 0.1*inch))
        
        # Overall Assessment Section
        elements.append(Paragraph("OVERALL ASSESSMENT", heading_style))
        assessment = analysis.get('overallAssessment', 'N/A')
        elements.append(Paragraph(assessment, normal_style))
        elements.append(Spacer(1, 0.15*inch))
        
        # Recommendations Section
        elements.append(Paragraph("CLINICAL RECOMMENDATIONS", heading_style))
        for rec in recommendations:
            elements.append(Paragraph(f"• {rec}", normal_style))
        elements.append(Spacer(1, 0.15*inch))
        
        # Care Plan Section (if available)
        if care_plan:
            elements.append(PageBreak())
            elements.append(Paragraph("PERSONALIZED PATIENT CARE PLAN", heading_style))
            # Split care plan into paragraphs and add them
            care_plan_paragraphs = care_plan.split('\n\n')
            for para in care_plan_paragraphs:
                if para.strip():
                    # Clean up markdown-style formatting
                    para_clean = para.replace('**', '').replace('*', '').strip()
                    if para_clean:
                        elements.append(Paragraph(para_clean, normal_style))
                        elements.append(Spacer(1, 0.1*inch))
            elements.append(Spacer(1, 0.15*inch))
        
        # Radiologist Information Section
        elements.append(Paragraph("RADIOLOGIST INFORMATION", heading_style))
        rad_info = [
            ['Radiologist:', radiologist.get('name', 'N/A')],
            ['Specialty:', radiologist.get('specialty', 'N/A')],
            ['License:', radiologist.get('license', 'N/A')]
        ]
        rad_table = Table(rad_info, colWidths=[1.5*inch, 4.5*inch])
        rad_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#E6F0F7')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey)
        ]))
        elements.append(rad_table)
        
        # Build PDF
        doc.build(elements)
        
        # Prepare PDF for download
        pdf_buffer.seek(0)
        
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"brain_scan_report_{patient.get('mrn', 'report')}.pdf"
        )
    
    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        return jsonify({"error": f"Failed to generate PDF: {str(e)}"}), 500


@app.route('/analyze', methods=['POST'])
def analyze_image():
    """Analyze uploaded brain scan image using ML models"""
    try:
        import sys
        import uuid
        
        # Check if torch is available
        try:
            import torch
        except ImportError:
            return jsonify({
                "error": "PyTorch not installed. Please run: pip install torch torchvision opencv-python numpy Pillow tqdm scikit-learn"
            }), 500
        
        # Add Model directory to Python path
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'Model')
        model_dir = os.path.abspath(model_dir)
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)
        
        # Import ML inference function
        try:
            from src.inference import analyze_brain_scan
        except ImportError as e:
            return jsonify({
                "error": f"Failed to import ML inference module: {str(e)}. Make sure Model directory is accessible."
            }), 500
        
        # Get patient info from request (optional, for storing in DB)
        # Can come from JSON or FormData
        patient_info = {}
        if request.is_json:
            patient_info = request.json.get('patient_info', {})
        elif 'patient_info' in request.form:
            try:
                patient_info = json.loads(request.form.get('patient_info', '{}'))
            except:
                patient_info = {}
        
        # Get image from request
        image_id = None
        if 'file' in request.files:
            # Direct file upload
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No file provided"}), 400
            
            # Store image in database first
            file_data = file.read()
            filename = secure_filename(file.filename)
            image_id = None
            if db.store_image(filename, file_data, metadata=patient_info):
                # Get the last inserted image ID
                images = db.get_all_images(limit=1)
                if images:
                    image_id = images[0]['id']
            
            # Save temporarily for analysis
            temp_filename = f"temp_{uuid.uuid4().hex[:8]}_{filename}"
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
            with open(temp_path, 'wb') as f:
                f.write(file_data)
            
            img_path = os.path.abspath(temp_path)
        elif 'image_id' in request.json:
            # Get image from database
            image_id = request.json['image_id']
            image_data = db.get_image(image_id)
            
            if not image_data:
                return jsonify({"error": "Image not found"}), 404
            
            # Parse JSON data
            import json
            img_json = json.loads(image_data['image_data'])
            img_base64 = img_json['image_data']
            
            # Save temporarily
            temp_filename = f"temp_{uuid.uuid4().hex[:8]}.jpg"
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
            
            with open(temp_path, 'wb') as f:
                f.write(base64.b64decode(img_base64))
            
            img_path = os.path.abspath(temp_path)
        else:
            return jsonify({"error": "No file or image_id provided"}), 400
        
        # Set up model paths (relative to Model directory)
        model_base = os.path.join(model_dir, 'models')
        classifier_path = os.path.join(model_base, 'resnet18_alzheimer.pth')
        autoencoder_path = os.path.join(model_base, 'autoencoder.pth')
        progression_path = os.path.join(model_base, 'progression_vector.pt')
        output_dir = os.path.join(model_dir, 'outputs')
        
        # Change to Model directory for relative paths to work
        original_cwd = os.getcwd()
        os.chdir(model_dir)
        
        # Convert img_path to relative path from Model directory if needed
        if os.path.isabs(img_path):
            # If absolute, keep it as is
            analysis_img_path = img_path
        else:
            # If relative, make it relative to Model directory
            analysis_img_path = os.path.join(model_dir, img_path)
        
        try:
            # Run ML analysis
            results = analyze_brain_scan(
                img_path=analysis_img_path,
                classifier_path=classifier_path,
                autoencoder_path=autoencoder_path,
                progression_path=progression_path,
                output_dir=output_dir,
                alpha=0.5
            )
        finally:
            # Restore original working directory
            os.chdir(original_cwd)
        
        # Convert images to base64 for frontend
        def image_to_base64(image_path):
            # Handle both absolute and relative paths
            if not os.path.isabs(image_path):
                image_path = os.path.join(model_dir, image_path)
            if os.path.exists(image_path):
                with open(image_path, 'rb') as f:
                    img_data = f.read()
                    return base64.b64encode(img_data).decode('utf-8')
            return None
        
        # Generate care plan using Gemini
        print("🤖 Generating care plan with Gemini...")
        care_plan_data = {
            "current_risk_score": results["current_risk_score"],
            "current_prediction": results["current_prediction"],
            "future_risk_score": results["future_risk_score"],
            "future_prediction": results["future_prediction"],
            "current_regions": results.get("current_regions", []),
            "future_regions": results.get("future_regions", [])
        }
        care_plan = generate_care_plan(care_plan_data)
        
        # Store analysis results in database if image_id is available
        analysis_id = None
        if image_id:
            analysis_id = db.store_analysis_results(
                image_id=image_id,
                patient_info=patient_info,
                analysis_data=care_plan_data,
                care_plan=care_plan
            )
            print(f"💾 Stored analysis results in DB with ID: {analysis_id}")
        
        # Prepare response
        response_data = {
            "current_risk_score": results["current_risk_score"],
            "current_prediction": results["current_prediction"],
            "future_risk_score": results["future_risk_score"],
            "future_prediction": results["future_prediction"],
            "current_heatmap": image_to_base64(results["current_heatmap_path"]),
            "future_mri": image_to_base64(results["future_mri_path"]),
            "future_heatmap": image_to_base64(results["future_heatmap_path"]),
            "original_image": image_to_base64(img_path),
            "current_regions": results.get("current_regions", []),
            "future_regions": results.get("future_regions", []),
            "care_plan": care_plan,
            "analysis_id": analysis_id
        }
        
        # Clean up temp file
        if os.path.exists(img_path) and img_path.startswith(app.config['UPLOAD_FOLDER']):
            try:
                os.remove(img_path)
            except:
                pass
        
        return jsonify(response_data), 200
    
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback.print_exc()
        return jsonify({"error": f"Analysis failed: {error_msg}"}), 500


@app.route('/image-file/<path:filename>', methods=['GET'])
def serve_image_file(filename):
    """Serve generated image files from Model/outputs"""
    try:
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'Model')
        file_path = os.path.join(model_dir, 'outputs', filename)
        
        if os.path.exists(file_path) and os.path.commonpath([model_dir, file_path]) == model_dir:
            return send_file(file_path)
        else:
            return jsonify({"error": "File not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/generate-care-plan', methods=['POST'])
def generate_care_plan_endpoint():
    """Generate a clinical care plan using Google Gemini API"""
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['current_risk_score', 'current_prediction', 'future_risk_score', 'future_prediction']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Generate the care plan
        care_plan = generate_care_plan(data)
        
        return jsonify({
            "care_plan": care_plan,
            "status": "success"
        }), 200
    
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback.print_exc()
        return jsonify({"error": f"Failed to generate care plan: {error_msg}"}), 500


if __name__ == '__main__':
    app.run(debug=True)

