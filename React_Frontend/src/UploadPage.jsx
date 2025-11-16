import { useState } from "react";
import "./UploadPage.css";
import PatientInfoForm from "./PatientInfoForm";

export default function UploadPage({ onUploadComplete }) {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadStatus, setUploadStatus] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [uploadedImages, setUploadedImages] = useState([]);
  const [patientInfo, setPatientInfo] = useState(null);
  const [showForm, setShowForm] = useState(true);

  const handleFileSelect = (e) => {
    const files = Array.from(e.target.files);
    setSelectedFiles(files);
    setUploadStatus("");
  };

  const handlePatientInfoSubmit = (info) => {
    setPatientInfo(info);
    setShowForm(false);
  };

  const handleBackToForm = () => {
    setShowForm(true);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.currentTarget.classList.add("drag-over");
  };

  const handleDragLeave = (e) => {
    e.currentTarget.classList.remove("drag-over");
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.currentTarget.classList.remove("drag-over");
    const files = Array.from(e.dataTransfer.files);
    setSelectedFiles(files);
    setUploadStatus("");
  };

  const handleUpload = async () => {
    if (selectedFiles.length === 0) {
      setUploadStatus("Please select at least one image.");
      return;
    }

    setIsUploading(true);
    setUploadProgress(0);

    try {
      const formData = new FormData();
      selectedFiles.forEach((file) => {
        formData.append("files", file);
      });

      const xhr = new XMLHttpRequest();

      xhr.upload.addEventListener("progress", (e) => {
        if (e.lengthComputable) {
          const percentComplete = (e.loaded / e.total) * 100;
          setUploadProgress(percentComplete);
        }
      });

      xhr.addEventListener("load", () => {
        if (xhr.status === 200) {
          const response = JSON.parse(xhr.responseText);
          setUploadStatus("✅ Images uploaded successfully!");
          setUploadedImages(response.uploaded_images || []);
          setSelectedFiles([]);
          setUploadProgress(0);
          
          // Trigger processing page after a short delay
          setTimeout(() => {
            onUploadComplete(patientInfo);
          }, 1500);
        } else {
          setUploadStatus("❌ Upload failed. Please try again.");
        }
        setIsUploading(false);
      });

      xhr.addEventListener("error", () => {
        setUploadStatus("❌ Network error during upload.");
        setIsUploading(false);
      });

      xhr.open("POST", "http://127.0.0.1:5000/upload", true);
      xhr.send(formData);
    } catch (error) {
      setUploadStatus(`❌ Error: ${error.message}`);
      setIsUploading(false);
    }
  };

  return (
    <>
      {showForm && patientInfo === null ? (
        <PatientInfoForm
          onSubmit={handlePatientInfoSubmit}
          onCancel={() => window.history.back()}
          isLoading={false}
        />
      ) : (
        <div className="upload-page">
          <div className="upload-container">
            <div className="upload-header">
              <h1 className="mb-4">
                <span className="brain-emoji">🧠</span> Upload Brain Scan Images
              </h1>
              {patientInfo && (
                <div className="patient-info-summary">
                  <p>
                    <strong>Patient:</strong> {patientInfo.patient.name} (MRN:{" "}
                    {patientInfo.patient.mrn})
                  </p>
                  <p>
                    <strong>Doctor:</strong> {patientInfo.doctor.name}
                  </p>
                  <button
                    onClick={handleBackToForm}
                    className="btn btn-link-small"
                  >
                    ✎ Edit Info
                  </button>
                </div>
              )}
            </div>

        {/* Drag and Drop Area */}
        <div
          className="upload-area"
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <div className="upload-icon">📁</div>
          <p className="upload-text">
            Drag and drop your images here or click to browse
          </p>
          <input
            type="file"
            multiple
            accept="image/*"
            onChange={handleFileSelect}
            style={{ display: "none" }}
            id="file-input"
          />
          <label htmlFor="file-input" className="upload-button btn btn-primary">
            Choose Images
          </label>
        </div>

        {/* Selected Files List */}
        {selectedFiles.length > 0 && (
          <div className="selected-files mt-4">
            <h5>Selected Files ({selectedFiles.length})</h5>
            <div className="file-list">
              {selectedFiles.map((file, index) => (
                <div key={index} className="file-item">
                  <span className="file-name">{file.name}</span>
                  <span className="file-size">
                    {(file.size / 1024 / 1024).toFixed(2)} MB
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Upload Progress */}
        {isUploading && (
          <div className="progress mt-4" style={{ height: "25px" }}>
            <div
              className="progress-bar bg-success"
              role="progressbar"
              style={{ width: `${uploadProgress}%` }}
              aria-valuenow={uploadProgress}
              aria-valuemin="0"
              aria-valuemax="100"
            >
              {uploadProgress.toFixed(0)}%
            </div>
          </div>
        )}

        {/* Upload Status */}
        {uploadStatus && (
          <div
            className={`alert mt-4 ${
              uploadStatus.includes("✅")
                ? "alert-success"
                : uploadStatus.includes("❌")
                ? "alert-danger"
                : "alert-info"
            }`}
          >
            {uploadStatus}
          </div>
        )}

        {/* Upload Button */}
        {!isUploading && selectedFiles.length > 0 && (
          <div className="mt-4">
            <button
              onClick={handleUpload}
              className="btn btn-success btn-lg"
              disabled={isUploading}
            >
              Upload {selectedFiles.length} Image{selectedFiles.length !== 1 ? "s" : ""}
            </button>
          </div>
        )}

        {/* Uploaded Images Display */}
        {uploadedImages.length > 0 && (
          <div className="uploaded-images mt-5">
            <h4>Recently Uploaded</h4>
            <div className="images-grid">
              {uploadedImages.map((imgData, index) => (
                <div key={index} className="image-card">
                  <img src={`data:image/jpeg;base64,${imgData.data}`} alt={imgData.filename} />
                  <p className="image-name">{imgData.filename}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Demo Brain Image */}
        <div className="demo-image-section">
          <h3>📋 Demo Brain Scan Image</h3>
          <div className="demo-image-container">
            <img src="/MRI_of_Human_Brain.jpg" alt="Demo MRI Brain Scan" />
          </div>
        </div>
          </div>
        </div>
      )}
    </>
  );
}