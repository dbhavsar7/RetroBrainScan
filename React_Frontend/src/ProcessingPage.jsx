import { useState, useEffect } from "react";
import "./ProcessingPage.css";

export default function ProcessingPage({ onProcessingComplete, uploadedImages, patientInfo }) {
  const [progress, setProgress] = useState(0);
  const [processingTime, setProcessingTime] = useState(0);
  const [currentStep, setCurrentStep] = useState("Uploading Image");
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!uploadedImages || uploadedImages.length === 0) {
      setError("No images to analyze");
      return;
    }

    const startTime = Date.now();
    const UPDATE_INTERVAL = 100; // Update every 100ms
    
    // Simulate progress while waiting for ML analysis
    const progressTimer = setInterval(() => {
      const elapsed = Date.now() - startTime;
      setProcessingTime(Math.floor(elapsed / 1000));
      
      // Simulate progress (will be overridden when real analysis completes)
      const simulatedProgress = Math.min((elapsed / 30000) * 90, 90); // Max 90% until done
      setProgress(simulatedProgress);
      
      // Update step based on time
      if (elapsed < 5000) {
        setCurrentStep("Analyzing Image");
      } else if (elapsed < 20000) {
        setCurrentStep("Generating Heatmaps");
      } else {
        setCurrentStep("Generating Care Plan");
      }
    }, UPDATE_INTERVAL);

    // Call analyze endpoint
    const analyzeImage = async () => {
      try {
        // Get first uploaded image
        const firstImage = uploadedImages[0];
        
        // Convert base64 to blob
        const byteCharacters = atob(firstImage.data);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
          byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        const blob = new Blob([byteArray], { type: 'image/jpeg' });
        
        // Create FormData
        const formData = new FormData();
        formData.append('file', blob, firstImage.filename);
        
        // Add patient info as JSON if available
        if (patientInfo) {
          formData.append('patient_info', JSON.stringify(patientInfo));
        }
        
        // Call analyze endpoint
        const response = await fetch('http://127.0.0.1:5000/analyze', {
          method: 'POST',
          body: formData
        });
        
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || 'Analysis failed');
        }
        
        const results = await response.json();
        
        // Set progress to 100%
        setProgress(100);
        setCurrentStep("Complete");
        
        // Wait a moment for visual effect, then complete
        setTimeout(() => {
          clearInterval(progressTimer);
          onProcessingComplete(results);
        }, 500);
        
      } catch (err) {
        console.error('Analysis error:', err);
        setError(err.message || 'Failed to analyze image');
        clearInterval(progressTimer);
      }
    };

    analyzeImage();

    return () => clearInterval(progressTimer);
  }, [uploadedImages, onProcessingComplete]);

  return (
    <div className="processing-page">
      <div className="processing-container">
        <div className="processing-icon">
          <img 
            src="/RBS_Logo_T.png" 
            alt="RetroBrainScan Logo" 
            style={{ width: '150px', height: '150px' }}
          />
        </div>
        <h2 className="processing-title">Analyzing Brain Scan</h2>
        <p className="processing-subtitle">{error || "Running AI model analysis..."}</p>

        {/* Animated loading dots */}
        <div className="loading-dots">
          <span></span>
          <span></span>
          <span></span>
        </div>

        {/* Progress Bar */}
        <div className="processing-progress-wrapper">
          <div className="processing-progress-bar">
            <div
              className="processing-progress-fill"
              style={{ width: `${progress}%` }}
            ></div>
          </div>
          <p className="progress-text">{Math.round(progress)}%</p>
        </div>

        {/* Processing Steps */}
        <div className="processing-steps">
          <div className={`step ${currentStep === "Analyzing Image" ? "active" : progress >= 25 ? "completed" : ""}`}>
            <span className="step-number">1</span>
            <span className="step-text">Analyzing Image</span>
          </div>
          <div className={`step ${currentStep === "Generating Heatmaps" ? "active" : progress >= 50 ? "completed" : ""}`}>
            <span className="step-number">2</span>
            <span className="step-text">Generating Heatmaps</span>
          </div>
          <div className={`step ${currentStep === "Generating Care Plan" || currentStep === "Complete" ? "active" : progress >= 80 ? "completed" : ""}`}>
            <span className="step-number">3</span>
            <span className="step-text">Generating Care Plan</span>
          </div>
        </div>

        <p className="processing-time">Processing: {processingTime}s</p>
      </div>
    </div>
  );
}
