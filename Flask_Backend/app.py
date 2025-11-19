# Flask backend
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from db_utils import DatabaseManager
import os
import base64
import json
import re
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib import colors
from datetime import datetime

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")
    print("Continuing without .env file support...")

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
GOOGLE_AI_API_KEY = os.getenv('GOOGLE_AI_API_KEY')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash-exp')

if GEMINI_AVAILABLE:
    if GOOGLE_AI_API_KEY:
        genai.configure(api_key=GOOGLE_AI_API_KEY)
    else:
        print("Warning: GOOGLE_AI_API_KEY environment variable not set. Gemini features will be disabled.")
        GEMINI_AVAILABLE = False


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
    """Generate a clinical care plan using Google Gemini API"""
    if not GEMINI_AVAILABLE:
        return "Error: Google Gemini API is not available. Please install google-generativeai package."
    
    try:
        system_prompt = load_clinical_prompt()
        current_regions = analysis_results.get("current_regions", [])
        future_regions = analysis_results.get("future_regions", [])
        
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
        
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(full_prompt)
        
        return response.text
        
    except Exception as e:
        error_msg = f"Error generating care plan: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return f"Error: {error_msg}"


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
                if not allowed_file(file.filename):
                    print(f"File rejected: {file.filename} - not an allowed image format")
                    continue
                
                filename = secure_filename(file.filename)
                file_data = file.read()
                
                if len(file_data) == 0:
                    print(f"File rejected: {file.filename} - empty file")
                    continue
                
                success = db.store_image(filename, file_data)
                
                if success:
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    with open(filepath, 'wb') as f:
                        f.write(file_data)
                    
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
        
        patient = data.get('patient', {})
        scan_info = data.get('scanInfo', {})
        analysis = data.get('analysis', {})
        recommendations = data.get('recommendations', [])
        doctor = data.get('doctor', data.get('radiologist', {}))
        analysis_results = data.get('analysisResults', {})
        care_plan = data.get('care_plan', '')
        
        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(
            pdf_buffer,
            pagesize=letter,
            rightMargin=0.5*inch,
            leftMargin=0.5*inch,
            topMargin=0.5*inch,
            bottomMargin=0.5*inch
        )
        
        elements = []
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#0B3555'),
            spaceAfter=10,
            alignment=0,
            fontName='Helvetica-Bold'
        )
        
        title_style_centered = ParagraphStyle(
            'CustomTitleCentered',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#0B3555'),
            spaceAfter=10,
            alignment=1,
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
        
        logo_path = os.path.join(os.path.dirname(__file__), '..', 'Assets', 'RBS_Logo_T.png')
        if not os.path.exists(logo_path):
            logo_path = os.path.join(os.path.dirname(__file__), '..', '..', 'Assets', 'RBS_Logo_T.png')
        
        header_data = []
        if os.path.exists(logo_path):
            try:
                logo_img = Image(logo_path, width=1.5*inch, height=1.5*inch)
                header_data.append([logo_img, Paragraph("RetroBrainScan Analysis Report", title_style_centered)])
            except Exception as e:
                print(f"Warning: Could not load logo: {e}")
                header_data.append(['', Paragraph("RetroBrainScan Analysis Report", title_style_centered)])
        else:
            header_data.append(['', Paragraph("RetroBrainScan Analysis Report", title_style_centered)])
        
        header_table = Table(header_data, colWidths=[2*inch, 4*inch])
        header_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (0, 0), 'LEFT'),
            ('ALIGN', (1, 0), (1, 0), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (0, 0), 0),
            ('RIGHTPADDING', (0, 0), (0, 0), 0),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ]))
        elements.append(header_table)
        elements.append(Spacer(1, 0.2*inch))
        
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
        
        if analysis_results:
            elements.append(Paragraph("BRAIN SCAN IMAGES", heading_style))
            images_row = []
            
            if analysis_results.get('original_image'):
                img = add_base64_image(analysis_results['original_image'])
                if img:
                    images_row.append(img)
            
            if analysis_results.get('current_heatmap'):
                img = add_base64_image(analysis_results['current_heatmap'])
                if img:
                    images_row.append(img)
            
            if images_row:
                img_table = Table([images_row], colWidths=[2.5*inch, 2.5*inch])
                img_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ]))
                elements.append(img_table)
                elements.append(Spacer(1, 0.1*inch))
            
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
            try:
                image_path = os.path.join(os.path.dirname(__file__), 'static', 'images', 'MRI_of_Human_Brain.jpg')
                if os.path.exists(image_path):
                    img = Image(image_path, width=3.5*inch, height=3.5*inch)
                    elements.append(img)
                    elements.append(Spacer(1, 0.2*inch))
            except Exception as e:
                print(f"Warning: Could not load brain scan image: {e}")
        
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
        
        elements.append(Paragraph("ANALYSIS FINDINGS", heading_style))
        findings = analysis.get('findings', [])
        for finding in findings:
            finding_text = f"<b>{finding.get('title', 'N/A')} - {finding.get('status', 'N/A')}</b><br/>"
            finding_text += finding.get('description', 'N/A')
            elements.append(Paragraph(finding_text, normal_style))
            elements.append(Spacer(1, 0.1*inch))
        
        elements.append(Spacer(1, 0.1*inch))
        
        elements.append(Paragraph("OVERALL ASSESSMENT", heading_style))
        assessment = analysis.get('overallAssessment', 'N/A')
        elements.append(Paragraph(assessment, normal_style))
        elements.append(Spacer(1, 0.15*inch))
        
        elements.append(Paragraph("CLINICAL RECOMMENDATIONS", heading_style))
        for rec in recommendations:
            elements.append(Paragraph(f"• {rec}", normal_style))
        elements.append(Spacer(1, 0.15*inch))
        
        if care_plan:
            elements.append(PageBreak())
            elements.append(Paragraph("PERSONALIZED PATIENT CARE PLAN", heading_style))
            care_plan_clean = re.sub(r'=+\n?', '', care_plan)
            care_plan_clean = care_plan_clean.replace('**', '')
            lines = care_plan_clean.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                underline_patterns = [
                    r'^A\.\s+Immediate\s+\(0–3\s+months\)',
                    r'^B\.\s+Short\s+Term\s+\(3–12\s+months\)',
                    r'^C\.\s+Long\s+Term\s+\(1–3\s+years\)',
                    r'^D\.\s+Future\s+Projection\s+\(3–5\s+years\)',
                    r'^Worsens:',
                    r'^Stable:',
                    r'^What\s+Worsens:',
                    r'^What\s+Stays\s+Stable:',
                    r'"Red\s+Flag"\s+Indicators:',
                    r'^Red\s+Flag\s+Indicators:',
                ]
                
                # Check if line matches any underline pattern
                should_underline = False
                for pattern in underline_patterns:
                    if re.match(pattern, line, re.IGNORECASE):
                        should_underline = True
                        break
                
                # Check if line starts with a number (numbered bullet point)
                numbered_match = re.match(r'^(\d+\.?\s+)(.+)$', line)
                if numbered_match:
                    # Make the entire line bold
                    if should_underline:
                        para_text = f"<b><u>{line}</u></b>"
                    else:
                        para_text = f"<b>{line}</b>"
                    elements.append(Paragraph(para_text, normal_style))
                else:
                    # Regular paragraph
                    if should_underline:
                        para_text = f"<u>{line}</u>"
                    else:
                        # Clean up any remaining markdown
                        para_text = line.replace('*', '').strip()
                    if para_text:
                        elements.append(Paragraph(para_text, normal_style))
                elements.append(Spacer(1, 0.1*inch))
            elements.append(Spacer(1, 0.15*inch))
        
        elements.append(Paragraph("RADIOLOGIST INFORMATION", heading_style))
        rad_info = [
            ['Radiologist:', doctor.get('name', 'N/A')],
            ['Specialty:', doctor.get('specialty', 'N/A')],
            ['License:', doctor.get('license', 'N/A')]
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
        
        doc.build(elements)
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
        
        try:
            import torch
        except ImportError:
            return jsonify({
                "error": "PyTorch not installed. Please run: pip install torch torchvision opencv-python numpy Pillow tqdm scikit-learn"
            }), 500
        
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'Model')
        model_dir = os.path.abspath(model_dir)
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)
        
        try:
            from src.inference import analyze_brain_scan
        except ImportError as e:
            return jsonify({
                "error": f"Failed to import ML inference module: {str(e)}. Make sure Model directory is accessible."
            }), 500
        
        patient_info = {}
        if request.is_json:
            patient_info = request.json.get('patient_info', {})
        elif 'patient_info' in request.form:
            try:
                patient_info = json.loads(request.form.get('patient_info', '{}'))
            except:
                patient_info = {}
        
        image_id = None
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No file provided"}), 400
            
            file_data = file.read()
            filename = secure_filename(file.filename)
            if db.store_image(filename, file_data, metadata=patient_info):
                images = db.get_all_images(limit=1)
                if images:
                    image_id = images[0]['id']
            
            temp_filename = f"temp_{uuid.uuid4().hex[:8]}_{filename}"
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
            with open(temp_path, 'wb') as f:
                f.write(file_data)
            
            img_path = os.path.abspath(temp_path)
        elif 'image_id' in request.json:
            image_id = request.json['image_id']
            image_data = db.get_image(image_id)
            
            if not image_data:
                return jsonify({"error": "Image not found"}), 404
            
            img_json = json.loads(image_data['image_data'])
            img_base64 = img_json['image_data']
            
            temp_filename = f"temp_{uuid.uuid4().hex[:8]}.jpg"
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
            
            with open(temp_path, 'wb') as f:
                f.write(base64.b64decode(img_base64))
            
            img_path = os.path.abspath(temp_path)
        else:
            return jsonify({"error": "No file or image_id provided"}), 400
        
        model_base = os.path.join(model_dir, 'models')
        classifier_path = os.path.join(model_base, 'resnet18_alzheimer.pth')
        autoencoder_path = os.path.join(model_base, 'autoencoder.pth')
        progression_path = os.path.join(model_base, 'progression_vector.pt')
        output_dir = os.path.join(model_dir, 'outputs')
        
        original_cwd = os.getcwd()
        os.chdir(model_dir)
        
        if os.path.isabs(img_path):
            analysis_img_path = img_path
        else:
            analysis_img_path = os.path.join(model_dir, img_path)
        
        try:
            results = analyze_brain_scan(
                img_path=analysis_img_path,
                classifier_path=classifier_path,
                autoencoder_path=autoencoder_path,
                progression_path=progression_path,
                output_dir=output_dir,
                alpha=0.5
            )
        finally:
            os.chdir(original_cwd)
        
        def image_to_base64(image_path):
            if not os.path.isabs(image_path):
                image_path = os.path.join(model_dir, image_path)
            if os.path.exists(image_path):
                with open(image_path, 'rb') as f:
                    img_data = f.read()
                    return base64.b64encode(img_data).decode('utf-8')
            return None
        
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
        
        analysis_id = None
        if image_id:
            analysis_id = db.store_analysis_results(
                image_id=image_id,
                patient_info=patient_info,
                analysis_data=care_plan_data,
                care_plan=care_plan
            )
        
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
        
        required_fields = ['current_risk_score', 'current_prediction', 'future_risk_score', 'future_prediction']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
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

