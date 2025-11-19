# RetroBrainScan - Project Index
2nd Best Hack Award Winner - HACKRPI 2025

## Overview
RetroBrainScan is a full-stack medical imaging application for analyzing brain MRI scans to predict Alzheimer's disease progression. The system uses deep learning models to classify current brain state and generate future MRI predictions with risk assessment.

## Project Structure

```
RetroBrainScan/
├── Assets/                          # Logo and branding assets
│   ├── RBS Analysis.png
│   ├── RBS Login.png
│   ├── RBS Upload.png
│   ├── RBS_Logo_T.png
│   ├── RBS_Logo_White.png
│   ├── RBS_Logo.png
│   └── RetroBrainScan Report Demo.png
├── Flask_Backend/                   # Python Flask API server
│   ├── app.py                       # Main Flask application (520+ lines)
│   ├── db_utils.py                  # Database manager (180 lines)
│   ├── install_and_run.py           # Setup script (135 lines)
│   ├── read_DB.py                   # Database reading utility
│   ├── requirements.txt             # Python dependencies
│   ├── RetroBrainScanDB.db          # SQLite database
│   ├── OUTPUT_IMAGES/               # Temporary image storage
│   └── prompts/
│       └── clinical_prompts.txt     # Clinical prompt templates
├── Model/                           # Machine Learning models and training code
│   ├── data/raw/                    # Training/test datasets
│   │   ├── train/                   # ~10,240 training images (4 classes)
│   │   ├── test/                    # ~1,279 test images
│   │   └── dementia_dataset.csv     # Dataset metadata
│   ├── models/                      # Trained model checkpoints
│   │   ├── autoencoder.pth          # Autoencoder for MRI generation
│   │   ├── progression_vector.pt    # Disease progression vector
│   │   ├── resnet18_alzheimer.pth   # ResNet18 classifier
│   │   └── vae_conv.pth             # VAE model (alternative)
│   ├── outputs/                     # Generated images (heatmaps, future MRIs)
│   ├── src/                         # Source code (16 files)
│   │   ├── autoencoder.py           # Autoencoder architecture
│   │   ├── brain_regions.json       # Brain region definitions
│   │   ├── compute_progression_vector.py
│   │   ├── data.py                  # Data loading utilities
│   │   ├── future_generator.py      # Future MRI generation
│   │   ├── generate_future_mri.py   # Standalone future MRI script
│   │   ├── gradcam.py               # GradCAM visualization
│   │   ├── inference.py             # Main inference pipeline
│   │   ├── models.py                # Neural network models
│   │   ├── progression.py           # Disease progression computation
│   │   ├── region_detector.py       # Brain region detection
│   │   ├── run_gradcam.py           # Standalone GradCAM script
│   │   ├── train_autoencoder.py     # Autoencoder training
│   │   ├── train_classifier.py      # Classifier training
│   │   └── vae.py                   # Variational Autoencoder
│   └── notebooks/                   # Jupyter notebooks
├── React_Frontend/                  # React.js web application
│   ├── src/
│   │   ├── App.jsx                  # Main application component
│   │   ├── App.css                  # Main styles
│   │   ├── main.jsx                 # Entry point
│   │   ├── index.css                # Global styles
│   │   ├── UploadPage.jsx           # Image upload interface
│   │   ├── UploadPage.css
│   │   ├── ProcessingPage.jsx       # Analysis processing interface
│   │   ├── ProcessingPage.css
│   │   ├── DoctorReportPage.jsx     # Results display and report generation
│   │   ├── DoctorReportPage.css
│   │   ├── PatientInfoForm.jsx      # Patient information form
│   │   ├── PatientInfoForm.css
│   │   ├── MessageSender.jsx        # Test component
│   │   ├── brain_regions.json       # Brain region definitions
│   │   ├── region_detector.py       # Brain region detection (Python)
│   │   └── prompts/
│   │       └── clinical_prompts.txt
│   ├── public/
│   │   ├── RBS_Logo_T.png
│   │   └── vite.svg
│   ├── package.json                 # Node.js dependencies
│   ├── vite.config.js               # Vite configuration
│   ├── eslint.config.js             # ESLint configuration
│   └── index.html                   # HTML entry point
├── README.md                        # Main project documentation
└── .gitignore                       # Git ignore rules
```

---

## 1. Flask Backend (`Flask_Backend/`)

### Purpose
RESTful API server that handles image uploads, ML inference, PDF report generation, and Google Gemini AI integration.

### Key Files

#### `app.py` (520+ lines)
Main Flask application with endpoints:
- **`POST /upload`** - Upload brain scan images (stores in SQLite DB)
- **`GET /images`** - Retrieve all uploaded images
- **`GET /image/<id>`** - Get specific image by ID
- **`GET /stats`** - Database statistics
- **`POST /analyze`** - Run ML analysis on uploaded image
  - Generates current heatmap (GradCAM)
  - Generates future MRI prediction
  - Generates future heatmap
  - Computes risk scores (current & future)
- **`POST /generate-report`** - Generate PDF medical report with Gemini AI
- **`GET /image-file/<filename>`** - Serve generated images
- **`POST /message`** - Test endpoint

**Features:**
- Google Gemini AI integration for clinical report generation
- CORS enabled for React frontend
- File upload handling with size limits (50MB)
- Base64 image encoding/decoding
- PDF generation with reportlab

#### `db_utils.py` (180 lines)
SQLite database manager:
- `DatabaseManager` class
- Methods: `store_image()`, `get_image()`, `get_all_images()`, `delete_image()`, `get_database_stats()`
- Stores images as base64-encoded JSON in SQLite
- Metadata support for patient information

#### `install_and_run.py` (135 lines)
Setup script:
- Creates virtual environment
- Installs dependencies from `requirements.txt`
- Initializes SQLite database
- Creates OUTPUT_IMAGES folder
- Runs Flask app on port 5000

#### `requirements.txt`
Python dependencies:
- Flask==3.1.2, flask-cors==6.0.1
- reportlab==4.0.9 (PDF generation)
- torch>=2.0.0, torchvision>=0.15.0 (PyTorch)
- opencv-python>=4.8.0, numpy>=1.24.0, Pillow>=10.0.0
- tqdm>=4.66.0, scikit-learn>=1.3.0
- google-generativeai>=0.3.0 (Gemini AI)

#### `read_DB.py`
Utility script for reading and inspecting database contents

#### `RetroBrainScanDB.db`
SQLite database storing uploaded images and metadata

#### `OUTPUT_IMAGES/`
Temporary storage for uploaded and processed images (cleaned up after use)

---

## 2. Model (`Model/`)

### Purpose
Machine learning models for Alzheimer's classification and future MRI generation.

### Directory Structure

#### `data/raw/`
- **`train/`** - Training images organized by class:
  - Mild Impairment/
  - Moderate Impairment/
  - No Impairment/
  - Very Mild Impairment/
- **`test/`** - Test images organized by class
- **`dementia_dataset.csv`** - Dataset metadata

#### `models/` - Trained Model Checkpoints
- **`resnet18_alzheimer.pth`** - Trained ResNet18 classifier
  - Contains: `model_state_dict`, `class_names`
- **`autoencoder.pth`** - Trained autoencoder
  - Contains: `model_state_dict`, `latent_dim`, `img_size`
- **`progression_vector.pt`** - Disease progression vector
  - Contains: `progression_vector`, `latent_dim`, `img_size`
- **`vae_conv.pth`** - VAE model (alternative approach)

#### `outputs/`
Generated images from inference:
- Current and future GradCAM heatmaps
- Future MRI predictions
- Visualization outputs

### Key Source Files (`Model/src/`)

#### `inference.py` (184 lines)
**Main inference pipeline:**
- `analyze_brain_scan()` - Complete analysis workflow
  - Generates current GradCAM heatmap
  - Generates future MRI using autoencoder
  - Generates future GradCAM heatmap
  - Computes risk scores for current and future states
- `load_classifier()` - Load ResNet18 classifier
- `get_risk_score()` - Compute Alzheimer's risk score (1 - P(No Impairment))
- `preprocess_for_classifier()` - Image preprocessing for ResNet

#### `future_generator.py` (76 lines)
**Future MRI generation:**
- `generate_future_mri()` - Generate predicted future MRI scan
  - Loads autoencoder
  - Encodes current MRI to latent space
  - Applies progression vector (alpha-weighted)
  - Decodes to generate future MRI
  - Generates GradCAM for future MRI
- `load_autoencoder()` - Load trained autoencoder model
- `load_image_gray()` - Load and preprocess grayscale MRI

#### `autoencoder.py` (73 lines)
**Autoencoder architecture:**
- `Encoder` - CNN encoder (1→32→64→128→256 channels, 128×128→8×8)
- `Decoder` - Transposed CNN decoder (8×8→128×128)
- `Autoencoder` - Full autoencoder wrapper
- Latent dimension: 64 (default)

#### `models.py` (36 lines)
**Neural network models:**
- `AlzheimerResNet` - ResNet18-based classifier
  - Pretrained on ImageNet
  - Customizable number of classes
  - Option to freeze backbone
- `get_device()` - Device selection (MPS for Apple Silicon, CUDA for GPU, else CPU)

#### `gradcam.py` (218 lines)
**GradCAM visualization:**
- `GradCAM` class - Gradient-weighted Class Activation Mapping
- `generate_cam()` - Main entry point for heatmap generation
- `load_image()` - Load and preprocess MRI (grayscale→RGB, crop borders)
- `overlay_heatmap()` - Blend heatmap with original MRI
  - Masks to brain region only
  - Keeps top 20% activations
  - Gaussian smoothing
  - Alpha blending

#### `progression.py` (94 lines)
**Disease progression vector computation:**
- `compute_progression_vector()` - Compute progression direction in latent space
  - Extracts latents from "No Impairment" class
  - Extracts latents from "Moderate Impairment" class
  - Computes mean difference: `progression_vector = z_moderate - z_no`
  - Saves to `models/progression_vector.pt`

#### `train_classifier.py` (169 lines)
Training script for ResNet18 classifier:
- Data loading and augmentation
- Training loop with validation
- Model checkpointing
- Class balancing

#### `train_autoencoder.py` (71 lines)
Training script for autoencoder:
- Reconstruction loss (MSE)
- Training on grayscale MRI images
- Checkpoint saving

#### `train_vae.py` (16 lines)
Training script for VAE (alternative model)

#### `vae.py` (182 lines)
Variational Autoencoder implementation:
- Encoder with reparameterization trick
- Decoder
- KL divergence loss

#### `data.py`
Data loading utilities:
- Dataset classes
- Data augmentation
- Train/test splits

#### `region_detector.py`
Brain region detection utilities

#### `compute_progression_vector.py`
Standalone script to compute progression vector from trained autoencoder

#### `generate_future_mri.py`
Standalone script for future MRI generation

#### `run_gradcam.py` (8 lines)
Standalone script for GradCAM visualization

#### `brain_regions.json`
JSON file defining brain regions and their properties

---

## 3. React Frontend (`React_Frontend/`)

### Purpose
Modern web interface for uploading MRIs, viewing analysis results, and generating reports.

### Technology Stack
- **React 19.2.0** - UI framework
- **Vite 7.2.2** - Build tool and dev server
- **Bootstrap** - CSS framework (via CDN)
- **ESLint** - Code linting

### Key Files

#### `src/App.jsx` (106 lines)
Main application component:
- State management for page navigation
- Routes between: Upload → Processing → Report
- Navigation bar with logo
- Footer with HACKRPI 2025 branding
- Error handling and loading states

#### `src/UploadPage.jsx`
Image upload interface:
- Patient information form
- Multi-file image upload (drag & drop)
- File validation
- Uploads to `/upload` endpoint
- On success → navigates to ProcessingPage

#### `src/ProcessingPage.jsx`
Analysis processing interface:
- Shows uploaded images
- Calls `/analyze` endpoint
- Displays loading state with progress
- On completion → navigates to DoctorReportPage
- Error handling

#### `src/DoctorReportPage.jsx`
Results display and report generation:
- Displays current and future MRI predictions
- Shows risk scores and predictions
- Displays heatmaps (GradCAM visualizations)
- Patient information display
- Generate PDF report button (calls `/generate-report`)
- Side-by-side comparison view

#### `src/PatientInfoForm.jsx`
Patient information form component:
- Patient name, age, gender
- Medical history fields
- Form validation

#### `src/MessageSender.jsx`
Test component for API communication

#### `package.json`
Dependencies:
- react: ^19.2.0, react-dom: ^19.2.0
- vite: ^7.2.2, @vitejs/plugin-react: ^5.1.0
- eslint (dev dependencies)

#### `vite.config.js`
Vite configuration for React development

#### `index.html`
HTML entry point for the React application

---

## 4. Data Flow

### Upload Flow
1. User uploads MRI images via React frontend
2. Frontend sends POST to `/upload` with FormData
3. Flask saves images to SQLite DB (base64) and disk (OUTPUT_IMAGES)
4. Returns uploaded image metadata with IDs

### Analysis Flow
1. User triggers analysis on uploaded image
2. Frontend sends POST to `/analyze` with image_id or file
3. Flask calls `analyze_brain_scan()` from `inference.py`
4. ML pipeline:
   - Loads ResNet18 classifier
   - Preprocesses image (128×128 RGB)
   - Generates current GradCAM heatmap
   - Loads autoencoder
   - Encodes current MRI → latent space (64-dim)
   - Applies progression vector → future latent
   - Decodes → future MRI (128×128)
   - Generates future GradCAM heatmap
   - Computes risk scores for both states
5. Results returned as JSON with base64-encoded images
6. Frontend displays results on DoctorReportPage

### Report Generation Flow
1. User clicks "Generate Report" on DoctorReportPage
2. Frontend sends POST to `/generate-report` with patient data and analysis results
3. Flask uses Google Gemini AI to generate clinical insights
4. Flask uses reportlab to generate PDF with:
   - Patient information
   - Current and future MRI images
   - Risk scores and predictions
   - GradCAM heatmaps
   - AI-generated clinical recommendations
5. PDF returned as download

---

## 5. Machine Learning Pipeline

### Classification Model
- **Architecture**: ResNet18 (pretrained on ImageNet)
- **Classes**: 4 classes
  - No Impairment
  - Very Mild Impairment
  - Mild Impairment
  - Moderate Impairment
- **Input**: 128×128 RGB image (grayscale converted to 3-channel)
- **Output**: Class probabilities (softmax)
- **Risk Score**: `1 - P(No Impairment)` (0-1 scale)

### Future MRI Generation
- **Architecture**: Autoencoder (Encoder-Decoder CNN)
- **Latent Space**: 64-dimensional
- **Progression Vector**: Direction from "No Impairment" → "Moderate Impairment" in latent space
- **Method**: `z_future = z_current + alpha * progression_vector`
- **Alpha**: 0.5 (default, controls progression strength)
- **Output**: 128×128 grayscale future MRI prediction

### Visualization
- **GradCAM**: Highlights regions important for classification
- **Target Layer**: ResNet18 layer3 (good spatial resolution)
- **Processing**: 
  - Masked to brain region only
  - Top 20% activations
  - Gaussian smoothing (sigma=2)
  - Alpha-blended with original MRI (alpha=0.4)

---

## 6. Database Schema

### ImageData Table
```sql
CREATE TABLE ImageData (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    upload_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    image_data TEXT NOT NULL,  -- JSON string with base64 image
    metadata TEXT              -- JSON string with patient info
)
```

---

## 7. API Endpoints Summary

| Method | Endpoint | Purpose | Request Body | Response |
|--------|----------|---------|--------------|----------|
| POST | `/upload` | Upload MRI images | FormData (files, patient info) | JSON with image IDs |
| GET | `/images` | List all images | - | JSON array of images |
| GET | `/image/<id>` | Get specific image | - | JSON with image data |
| GET | `/stats` | Database statistics | - | JSON with counts |
| POST | `/analyze` | Run ML analysis | JSON (image_id or file) | JSON with results, base64 images |
| POST | `/generate-report` | Generate PDF report | JSON (patient data, results) | PDF file |
| GET | `/image-file/<filename>` | Serve image files | - | Image file |
| POST | `/message` | Test endpoint | JSON | Echo response |

---

## 8. Key Dependencies

### Backend (Python)
- Flask==3.1.2
- flask-cors==6.0.1
- PyTorch>=2.0.0
- torchvision>=0.15.0
- opencv-python>=4.8.0
- numpy>=1.24.0
- Pillow>=10.0.0
- reportlab==4.0.9
- google-generativeai>=0.3.0
- tqdm>=4.66.0
- scikit-learn>=1.3.0

### Frontend (Node.js)
- React 19.2.0
- React DOM 19.2.0
- Vite 7.2.2
- ESLint 9.39.1

---

## 9. Setup & Run

### Backend Setup
```bash
cd Flask_Backend
python install_and_run.py
```
This will:
- Create a virtual environment
- Install all dependencies
- Initialize the database
- Start Flask server on http://localhost:5000

### Frontend Setup
```bash
cd React_Frontend
npm install
npm run dev
```
This will:
- Install Node.js dependencies
- Start Vite dev server (usually http://localhost:5173)

### Access Application
Open browser to the frontend URL (typically http://localhost:5173/)

---

## 10. File Counts & Statistics

- **Backend Python files**: 4 main files (app.py, db_utils.py, install_and_run.py, read_DB.py)
- **Model source files**: 16 Python files
- **Frontend components**: 6 React components (JSX)
- **Trained models**: 4 checkpoint files
- **Training data**: ~10,240 training images (4 classes)
- **Test data**: ~1,279 test images
- **Asset files**: 7 image files (logos, screenshots)

---

## 11. Key Features

1. **Multi-class Alzheimer's Classification** - 4 severity levels using ResNet18
2. **Future MRI Prediction** - Generative autoencoder predicts disease progression
3. **Risk Score Calculation** - Quantifies Alzheimer's risk (0-1 scale)
4. **GradCAM Visualization** - Highlights important brain regions for diagnosis
5. **PDF Report Generation** - Professional medical reports with AI insights
6. **Image Database** - SQLite storage for uploaded scans and metadata
7. **RESTful API** - Clean separation between frontend and backend
8. **Google Gemini AI Integration** - AI-powered clinical report generation
9. **Responsive UI** - Modern React frontend with Bootstrap styling
10. **Multi-file Upload** - Support for multiple MRI scan uploads

---

## 12. Technical Notes

- Models support Apple Silicon (MPS) acceleration
- Images are stored as base64-encoded JSON in database
- Temporary files are cleaned up after analysis
- Unique filenames prevent conflicts in concurrent requests
- All paths are relative to Model directory during inference
- CORS enabled for cross-origin requests
- File size limit: 50MB per upload
- Supported image formats: PNG, JPG, JPEG, GIF, BMP, TIFF

---

## 13. Development Workflow

1. **Model Training**: Train models in `Model/src/` using training scripts
2. **Backend Development**: Modify `Flask_Backend/app.py` for API changes
3. **Frontend Development**: Modify React components in `React_Frontend/src/`
4. **Testing**: Use test endpoints and frontend interface
5. **Deployment**: Build frontend with `npm run build`, deploy Flask app

---

*Project: RetroBrainScan - HACKRPI 2025*
*Full-stack medical imaging application for Alzheimer's disease prediction*

