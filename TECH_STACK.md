# RetroBrainScan - Tech Stack & Workflow

## 🛠️ Complete Tech Stack

### **Frontend**
- **React 19.2.0** - Modern UI framework for building interactive user interfaces
- **Vite 7.2.2** - Fast build tool and development server
- **Bootstrap** - CSS framework for responsive design
- **JavaScript (ES6+)** - Modern JavaScript features

### **Backend**
- **Flask 3.1.2** - Lightweight Python web framework
- **Flask-CORS 6.0.1** - Cross-Origin Resource Sharing support
- **Python 3.x** - Backend programming language

### **Machine Learning & AI**
- **PyTorch ≥2.0.0** - Deep learning framework
- **Torchvision ≥0.15.0** - Computer vision utilities and pre-trained models
- **ResNet18** - Pre-trained CNN architecture for image classification
- **Custom Autoencoder** - Neural network for future MRI generation
- **GradCAM** - Gradient-weighted Class Activation Mapping for visualization
- **NumPy ≥1.24.0** - Numerical computing library
- **OpenCV-Python ≥4.8.0** - Computer vision and image processing
- **Pillow ≥10.0.0** - Image processing library
- **scikit-learn ≥1.3.0** - Machine learning utilities

### **AI/LLM Integration**
- **Google Generative AI (Gemini 2.0 Flash)** - Large Language Model for clinical care plan generation
- **google-generativeai 0.8.5** - Official Python SDK for Google Gemini API

### **Database**
- **SQLite3** - Lightweight relational database for storing:
  - Uploaded MRI images
  - Patient information
  - Analysis results
  - Generated care plans

### **PDF Generation**
- **ReportLab 4.0.9** - Python library for generating professional PDF reports

### **Development Tools**
- **tqdm ≥4.66.0** - Progress bars for long-running operations
- **ESLint** - JavaScript code linting
- **Node.js & npm** - Package management for frontend

### **Deployment & Infrastructure**
- **Virtual Environment (venv)** - Python environment isolation
- **RESTful API** - Standard API architecture
- **Base64 Encoding** - Image data encoding for transmission

---

## 🔄 Complete Workflow

### **Phase 1: Patient Information Collection**
1. User navigates to the **Patient Information Form**
2. Fills in patient details (name, age, MRN, DOB, etc.)
3. Fills in radiologist information (name, specialty, license)
4. Option to use **Autofill** button for quick testing
5. Clicks **"Continue to Upload"** button

### **Phase 2: Image Upload**
1. User uploads brain MRI scan image(s) via drag-and-drop or file browser
2. Images are:
   - Validated (format, size)
   - Stored in **SQLite database** (base64 encoded)
   - Saved to disk as backup
3. Upload progress is displayed with real-time feedback
4. On successful upload, proceeds to analysis

### **Phase 3: ML Analysis Pipeline**

#### **3.1 Current State Analysis**
1. **Image Preprocessing**
   - Convert to grayscale
   - Resize to 128×128 pixels
   - Normalize pixel values
   - Convert to RGB format for ResNet18

2. **Classification (ResNet18)**
   - Load pre-trained ResNet18 model
   - Classify into 4 categories:
     - No Impairment
     - Very Mild Impairment
     - Mild Impairment
     - Moderate Impairment
   - Calculate **current risk score**: `1 - P(No Impairment)`

3. **GradCAM Visualization**
   - Generate Class Activation Map (CAM) for current MRI
   - Highlight brain regions important for classification
   - Create heatmap overlay on original image
   - Save as **current_heatmap.png**

4. **Brain Region Detection**
   - Divide CAM into 3×3 spatial zones
   - Map zones to anatomical brain regions:
     - Frontal Lobe, Temporal Lobe, Parietal Lobe
     - Hippocampus, Amygdala, Precuneus
     - Posterior Cingulate Cortex, Thalamus, Basal Ganglia
   - Extract top 2 most activated regions

#### **3.2 Future State Prediction**
1. **Autoencoder Processing**
   - Load trained autoencoder model
   - Encode current MRI to 64-dimensional latent space
   - Apply **progression vector** (learned direction from "No Impairment" → "Moderate Impairment")
   - Shift latent representation: `z_future = z_current + α × progression_vector`
   - Decode to generate **predicted future MRI**

2. **Future Classification**
   - Classify predicted future MRI using same ResNet18 model
   - Calculate **future risk score**
   - Generate **future prediction** classification

3. **Future GradCAM**
   - Generate CAM for predicted future MRI
   - Create **future_heatmap.png**
   - Extract future brain regions

### **Phase 4: AI-Powered Care Plan Generation**
1. **Data Preparation**
   - Format analysis results:
     - Current & future risk scores
     - Current & future classifications
     - Current & future brain regions
   - Structure data as JSON

2. **Google Gemini API Call**
   - Load clinical prompt template (acts as neuroradiologist)
   - Send formatted data to **Gemini 2.0 Flash** model
   - Generate comprehensive clinical care plan

3. **Care Plan Sections Generated:**
   - **IMPRESSION** - Diagnosis-style summary
   - **DETAILED NEUROANATOMICAL INTERPRETATION** - Region-by-region analysis
   - **RISK TRAJECTORY SUMMARY** - Current vs future comparison
   - **PREDICTED FUNCTIONAL IMPACT** - Expected symptoms by region
   - **PERSONALIZED 5-YEAR CARE PLAN**:
     - Immediate (0-3 months)
     - Short Term (3-12 months)
     - Long Term (1-3 years)
     - Future Projection (3-5 years)
   - **PRESENT-TO-FUTURE DIFFERENCE SUMMARY** - What worsens, stays stable, red flags

### **Phase 5: Data Storage**
1. Store complete analysis in **SQLite database**:
   - Image ID (foreign key)
   - Patient information (JSON)
   - Current risk score & prediction
   - Future risk score & prediction
   - Current & future brain regions (JSON arrays)
   - Generated care plan (text)
   - Timestamp

### **Phase 6: Results Display**
1. **Doctor Report Page** shows:
   - **4 Brain Scan Images**:
     - Original Current MRI
     - Current Heatmap (GradCAM)
     - Predicted Future MRI
     - Future Heatmap (GradCAM)
   - **Analysis Findings**:
     - Current classification & risk score
     - Future prediction & risk score
   - **Brain Regions**:
     - Current affected regions
     - Future affected regions
   - **Personalized Care Plan**:
     - Full clinical report from Gemini
     - All 6 sections formatted

### **Phase 7: PDF Report Generation**
1. User clicks **"Download PDF"** button
2. **ReportLab** generates professional PDF containing:
   - Patient information
   - Scan information
   - All 4 brain scan images (2×2 grid)
   - Analysis findings
   - Overall assessment
   - Clinical recommendations
   - **Full care plan** (separate page)
   - Radiologist information
3. PDF is downloaded to user's device

---

## 📊 Data Flow Diagram

```
User Input
    ↓
[Patient Info Form] → SQLite DB
    ↓
[Image Upload] → SQLite DB (ImageData table)
    ↓
[ML Analysis Pipeline]
    ├─→ ResNet18 Classifier → Current Risk Score & Prediction
    ├─→ GradCAM → Current Heatmap
    ├─→ Region Detector → Current Brain Regions
    ├─→ Autoencoder → Future MRI Generation
    ├─→ ResNet18 Classifier → Future Risk Score & Prediction
    ├─→ GradCAM → Future Heatmap
    └─→ Region Detector → Future Brain Regions
    ↓
[Google Gemini API] → Care Plan Generation
    ↓
[SQLite DB] → AnalysisResults table (stores everything)
    ↓
[Frontend Display] → Doctor Report Page
    ↓
[PDF Generation] → ReportLab → Download
```

---

## 🎯 Key Features

1. **Automated Analysis** - Complete ML pipeline from image to insights
2. **Future Prediction** - AI-generated progression modeling
3. **Visual Explanations** - GradCAM heatmaps show model reasoning
4. **Anatomical Mapping** - Identifies specific brain regions affected
5. **AI-Generated Care Plans** - Professional clinical reports via Gemini
6. **Data Persistence** - All results stored in database
7. **Professional PDFs** - Medical-grade report generation
8. **Responsive UI** - Modern, user-friendly interface

---

## 🔬 Model Architecture Details

### **ResNet18 Classifier**
- **Input**: 128×128 RGB images
- **Architecture**: Pre-trained ResNet18 (ImageNet weights)
- **Output**: 4-class classification (Alzheimer's severity levels)
- **Purpose**: Risk assessment and classification

### **Autoencoder**
- **Encoder**: 4-layer CNN (1→32→64→128→256 channels)
- **Latent Space**: 64 dimensions
- **Decoder**: 4-layer transposed CNN (256→128→64→32→1 channels)
- **Purpose**: Future MRI generation via latent space manipulation

### **Progression Vector**
- **Method**: Mean difference in latent space
- **Calculation**: `z_moderate - z_no_impairment`
- **Usage**: Applied with alpha factor (0.5) to predict progression
- **Purpose**: Disease progression modeling

---

## 📦 API Endpoints

- `POST /upload` - Upload MRI images
- `GET /images` - Retrieve all uploaded images
- `GET /image/<id>` - Get specific image
- `POST /analyze` - Run complete ML analysis + care plan generation
- `POST /generate-report` - Generate PDF report
- `POST /generate-care-plan` - Generate care plan only (standalone)
- `GET /stats` - Database statistics

---

## 🚀 Innovation Highlights

1. **End-to-End Pipeline** - From image upload to clinical report
2. **Progressive Disease Modeling** - Predicts future brain state
3. **Explainable AI** - GradCAM visualizations show model focus
4. **Clinical Integration** - LLM generates professional medical reports
5. **Complete Data Management** - Persistent storage of all analyses
6. **Production-Ready** - Error handling, validation, professional UI

---

*Built for HACKRPI 2025*

