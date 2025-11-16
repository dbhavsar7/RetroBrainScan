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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from datetime import datetime

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
            textColor=colors.HexColor('#667eea'),
            spaceAfter=10,
            alignment=1,  # Center
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#667eea'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold',
            borderPadding=5,
            borderColor=colors.HexColor('#667eea'),
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
        patient_info = [
            ['Name:', patient.get('name', 'N/A')],
            ['Age:', str(patient.get('age', 'N/A'))],
            ['Gender:', patient.get('gender', 'N/A')],
            ['MRN:', patient.get('mrn', 'N/A')],
            ['DOB:', patient.get('dateOfBirth', 'N/A')],
            ['Hospital:', patient.get('hospital', 'N/A')]
        ]
        patient_table = Table(patient_info, colWidths=[1.5*inch, 4.5*inch])
        patient_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8e9ff')),
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
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8e9ff')),
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
        
        # Radiologist Information Section
        elements.append(Paragraph("RADIOLOGIST INFORMATION", heading_style))
        rad_info = [
            ['Radiologist:', radiologist.get('name', 'N/A')],
            ['Specialty:', radiologist.get('specialty', 'N/A')],
            ['License:', radiologist.get('license', 'N/A')]
        ]
        rad_table = Table(rad_info, colWidths=[1.5*inch, 4.5*inch])
        rad_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8e9ff')),
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


if __name__ == '__main__':
    app.run(debug=True)

