import { useState, useEffect } from "react";
import "./ProcessingPage.css";

export default function ProcessingPage({ onProcessingComplete }) {
  const [progress, setProgress] = useState(0);
  const [processingTime, setProcessingTime] = useState(0);

  useEffect(() => {
    // Simulate 5-second processing with progress bar
    const PROCESSING_DURATION = 5000; // 5 seconds
    const UPDATE_INTERVAL = 50; // Update every 50ms

    const startTime = Date.now();
    
    const timer = setInterval(() => {
      const elapsed = Date.now() - startTime;
      const newProgress = Math.min((elapsed / PROCESSING_DURATION) * 100, 100);
      
      setProgress(newProgress);
      setProcessingTime(Math.floor(elapsed / 1000));

      // When processing complete
      if (newProgress >= 100) {
        clearInterval(timer);
        setTimeout(() => {
          onProcessingComplete();
        }, 500); // Small delay for visual effect
      }
    }, UPDATE_INTERVAL);

    return () => clearInterval(timer);
  }, [onProcessingComplete]);

  return (
    <div className="processing-page">
      <div className="processing-container">
        <div className="processing-icon">🧠</div>
        <h2 className="processing-title">Analyzing Brain Scan</h2>
        <p className="processing-subtitle">Running AI model analysis...</p>

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
          <div className={`step ${progress >= 20 ? "active" : ""}`}>
            <span className="step-number">1</span>
            <span className="step-text">Uploading Image</span>
          </div>
          <div className={`step ${progress >= 50 ? "active" : ""}`}>
            <span className="step-number">2</span>
            <span className="step-text">Processing</span>
          </div>
          <div className={`step ${progress >= 80 ? "active" : ""}`}>
            <span className="step-number">3</span>
            <span className="step-text">Generating Report</span>
          </div>
        </div>

        <p className="processing-time">Processing: {processingTime}s</p>
      </div>
    </div>
  );
}
