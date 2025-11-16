import { useState } from "react";
import "./UploadPage.css";
import PatientInfoForm from "./PatientInfoForm";

export default function UploadPage({ onUploadComplete, autoFillAndSubmit, onAutoFillComplete }) {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [uploadProgress, setUploadProgress] = useState({}); // Object to track progress per file
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
    // Initialize progress for each file
    const initialProgress = {};
    selectedFiles.forEach((file) => {
      initialProgress[file.name] = 0;
    });
    setUploadProgress(initialProgress);
    setUploadStatus("Uploading images...");

    // Simulate progress animation to 90% over 2.5 seconds for all files
    let simulatedProgress = 0;
    const progressInterval = setInterval(() => {
      simulatedProgress += 1.8; // Increment by 1.8% every ~45ms to reach 90% in ~2.5s
      if (simulatedProgress >= 90) {
        simulatedProgress = 90;
        clearInterval(progressInterval);
      }
      // Update progress for all files
      const newProgress = {};
      selectedFiles.forEach((file) => {
        newProgress[file.name] = simulatedProgress;
      });
      setUploadProgress(newProgress);
    }, 45);

    try {
      const formData = new FormData();
      selectedFiles.forEach((file) => {
        formData.append("files", file);
      });

      const xhr = new XMLHttpRequest();

      xhr.upload.addEventListener("progress", (e) => {
        if (e.lengthComputable) {
          const percentComplete = (e.loaded / e.total) * 100;
          // Use actual upload progress if it's ahead of simulated progress
          if (percentComplete > simulatedProgress && percentComplete < 90) {
            const newProgress = {};
            selectedFiles.forEach((file) => {
              newProgress[file.name] = percentComplete;
            });
            setUploadProgress(newProgress);
            simulatedProgress = percentComplete;
          }
        }
      });

      xhr.addEventListener("load", () => {
        clearInterval(progressInterval);
        
        if (xhr.status === 200) {
          const response = JSON.parse(xhr.responseText);
          
          // Complete the progress bar to 100% for all files
          const completeProgress = {};
          selectedFiles.forEach((file) => {
            completeProgress[file.name] = 100;
          });
          setUploadProgress(completeProgress);
          setUploadStatus("✅ Images uploaded successfully!");
          setUploadedImages(response.uploaded_images || []);
          
          // Wait a moment to show 100%, then move to processing
          setTimeout(() => {
            setSelectedFiles([]);
            setUploadProgress({});
            onUploadComplete(patientInfo, response.uploaded_images || []);
          }, 800);
        } else {
          setUploadStatus("❌ Upload failed. Please try again.");
          setIsUploading(false);
          setUploadProgress({});
        }
      });

      xhr.addEventListener("error", () => {
        clearInterval(progressInterval);
        setUploadStatus("❌ Network error during upload.");
        setIsUploading(false);
        setUploadProgress({});
      });

      xhr.open("POST", "http://127.0.0.1:5000/upload", true);
      xhr.send(formData);
    } catch (error) {
      setUploadStatus(`❌ Error: ${error.message}`);
      setIsUploading(false);
      setUploadProgress({});
    }
  };

  return (
    <>
      {showForm && patientInfo === null ? (
        <PatientInfoForm
          onSubmit={handlePatientInfoSubmit}
          onCancel={() => window.history.back()}
          isLoading={false}
          autoFillAndSubmit={autoFillAndSubmit}
          onAutoFillComplete={onAutoFillComplete}
        />
      ) : (
        <div className="upload-page">
          <div className="upload-container">
            <div className="upload-header">
              <h1 className="mb-4">
                Upload Brain Scan Images
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
            <div className="selected-files-header">
              <h5>Selected Files</h5>
              <button
                className="btn-clear-files"
                onClick={() => setSelectedFiles([])}
                title="Clear all selected files"
              >
                🗑️
              </button>
            </div>
            <div className="file-list">
              {selectedFiles.map((file, index) => {
                const fileSizeMB = file.size / 1024 / 1024;
                const fileSizeKB = file.size / 1024;
                const displaySize = fileSizeMB >= 1 
                  ? `${fileSizeMB.toFixed(2)} MB` 
                  : `${fileSizeKB.toFixed(2)} KB`;
                
                const progress = uploadProgress[file.name] || 0;
                const isUploadingFile = isUploading && progress > 0;
                
                return (
                  <div key={index} className="file-item">
                    {isUploadingFile ? (
                      <>
                        <div 
                          className="file-progress-bar progress-bar-upload"
                          style={{ 
                            position: 'absolute',
                            left: 0,
                            top: 0,
                            height: '100%',
                            width: `${progress}%`,
                            transition: 'width 0.4s ease',
                            zIndex: 0
                          }}
                        />
                        <div style={{ position: 'relative', zIndex: 1, display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%', padding: '10px' }}>
                          <span className="file-name" style={{ color: progress === 100 ? 'white' : '#333' }}>
                            {file.name}
                          </span>
                          <span className="file-size" style={{ color: progress === 100 ? 'white' : '#999' }}>
                            {progress.toFixed(0)}%
                          </span>
                        </div>
                      </>
                    ) : (
                      <>
                        <span className="file-name">{file.name}</span>
                        <span className="file-size">
                          {displaySize}
                        </span>
                      </>
                    )}
                  </div>
                );
              })}
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
        {/* <div className="demo-image-section">
          <h3>📋 Demo Brain Scan Image</h3>
          <div className="demo-image-container">
            <img src="/MRI_of_Human_Brain.jpg" alt="Demo MRI Brain Scan" />
          </div>
        </div> */}
          </div>
        </div>
      )}
    </>
  );
}