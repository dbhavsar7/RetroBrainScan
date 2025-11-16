import { useState } from "react";
import "./DoctorReportPage.css";

const DUMMY_REPORT_DATA = {
  patient: {
    name: "John Doe",
    age: 45,
    gender: "Male",
    mrn: "MRN-2025-001234",
    dateOfBirth: "1979-06-15",
    hospital: "General Medical Center",
  },
  scanInfo: {
    scanType: "MRI Brain",
    scanDate: "2025-11-15",
    scanTime: "14:30:00",
    scanDuration: "32 minutes",
    scannerModel: "Siemens 3.0T",
  },
  analysis: {
    findings: [
      {
        title: "Brain Structure",
        status: "Normal",
        description:
          "No significant abnormalities detected in brain structure. Ventricles and sulci appear normal for age.",
        confidence: 98,
      },
      {
        title: "White Matter",
        status: "Mild Changes",
        description:
          "Mild periventricular white matter changes consistent with age. No acute findings.",
        confidence: 96,
      },
      {
        title: "Vasculature",
        status: "Normal",
        description:
          "Intracranial arterial and venous vasculature is patent. No evidence of aneurysm or stenosis.",
        confidence: 99,
      },
      {
        title: "Lesions",
        status: "Negative",
        description: "No acute intracranial lesions or masses identified.",
        confidence: 97,
      },
    ],
    overallAssessment:
      "The imaging study demonstrates no acute intracranial abnormalities. Mild age-related changes noted. Clinical correlation recommended.",
  },
  recommendations: [
    "Follow-up imaging in 12 months if clinically indicated",
    "Consider neurology consultation for persistent symptoms",
    "Recommend lifestyle modifications for vascular health",
  ],
  radiologist: {
    name: "Dr. Sarah Johnson, MD",
    specialty: "Neuroradiology",
    license: "License #NR-45628",
  },
};

export default function DoctorReportPage({ onBackClick }) {
  const [showRawData, setShowRawData] = useState(false);
  const [imageSrc, setImageSrc] = useState("/MRI_of_Human_Brain.jpg");
  const [imageError, setImageError] = useState(false);

  const handleDownloadPDF = async () => {
    try {
      const response = await fetch(
        "http://127.0.0.1:5000/generate-report",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(DUMMY_REPORT_DATA),
        }
      );

      if (!response.ok) {
        alert("Failed to generate PDF");
        return;
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `brain_scan_report_${DUMMY_REPORT_DATA.patient.mrn}.pdf`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error("Error downloading PDF:", error);
      alert("Error downloading PDF");
    }
  };

  const getStatusClass = (status) => {
    switch (status.toLowerCase()) {
      case "normal":
        return "status-normal";
      case "mild changes":
        return "status-mild";
      case "negative":
        return "status-normal";
      case "abnormal":
        return "status-abnormal";
      default:
        return "status-unknown";
    }
  };

  return (
    <div className="doctor-report-page">
      <div className="report-container">
        {/* Header */}
        <div className="report-header">
          <div className="header-content">
            <h1>🧠 Brain Scan Analysis Report</h1>
            <p className="report-id">Report ID: {DUMMY_REPORT_DATA.patient.mrn}</p>
          </div>
          <div className="header-actions">
            <button className="btn btn-primary" onClick={handleDownloadPDF}>
              📥 Download PDF
            </button>
            <button className="btn btn-secondary" onClick={onBackClick}>
              ← Back to Upload
            </button>
          </div>
        </div>

        {/* Brain Scan Image */}
        <section className="report-section brain-scan-section">
          <div className="brain-scan-image-container">
            {!imageError ? (
              <img 
                src={imageSrc} 
                alt="MRI Brain Scan" 
                className="brain-scan-image"
                onError={() => {
                  // Try fallback source from backend static folder
                  if (imageSrc === "/MRI_of_Human_Brain.jpg") {
                    setImageSrc("http://127.0.0.1:5000/static/images/MRI_of_Human_Brain.jpg");
                  } else {
                    setImageError(true);
                  }
                }}
              />
            ) : (
              <div style={{ 
                padding: "40px", 
                textAlign: "center", 
                color: "#999",
                fontSize: "0.9rem"
              }}>
                <p>📁 MRI Brain Scan Image</p>
                <p style={{ fontSize: "0.8rem", marginTop: "10px" }}>
                  (Image not found - place MRI_of_Human_Brain.jpg in public folder)
                </p>
              </div>
            )}
          </div>
        </section>

        {/* Patient Information */}
        <section className="report-section patient-info">
          <h2>Patient Information</h2>
          <div className="info-grid">
            <div className="info-item">
              <span className="label">Name:</span>
              <span className="value">{DUMMY_REPORT_DATA.patient.name}</span>
            </div>
            <div className="info-item">
              <span className="label">Age:</span>
              <span className="value">{DUMMY_REPORT_DATA.patient.age}</span>
            </div>
            <div className="info-item">
              <span className="label">Gender:</span>
              <span className="value">{DUMMY_REPORT_DATA.patient.gender}</span>
            </div>
            <div className="info-item">
              <span className="label">MRN:</span>
              <span className="value">{DUMMY_REPORT_DATA.patient.mrn}</span>
            </div>
            <div className="info-item">
              <span className="label">Date of Birth:</span>
              <span className="value">{DUMMY_REPORT_DATA.patient.dateOfBirth}</span>
            </div>
            <div className="info-item">
              <span className="label">Hospital:</span>
              <span className="value">{DUMMY_REPORT_DATA.patient.hospital}</span>
            </div>
          </div>
        </section>

        {/* Scan Information */}
        <section className="report-section scan-info">
          <h2>Scan Information</h2>
          <div className="info-grid">
            <div className="info-item">
              <span className="label">Scan Type:</span>
              <span className="value">{DUMMY_REPORT_DATA.scanInfo.scanType}</span>
            </div>
            <div className="info-item">
              <span className="label">Scan Date:</span>
              <span className="value">{DUMMY_REPORT_DATA.scanInfo.scanDate}</span>
            </div>
            <div className="info-item">
              <span className="label">Scan Time:</span>
              <span className="value">{DUMMY_REPORT_DATA.scanInfo.scanTime}</span>
            </div>
            <div className="info-item">
              <span className="label">Duration:</span>
              <span className="value">{DUMMY_REPORT_DATA.scanInfo.scanDuration}</span>
            </div>
            <div className="info-item">
              <span className="label">Scanner Model:</span>
              <span className="value">{DUMMY_REPORT_DATA.scanInfo.scannerModel}</span>
            </div>
          </div>
        </section>

        {/* Analysis Findings */}
        <section className="report-section findings">
          <h2>Analysis Findings</h2>
          <div className="findings-list">
            {DUMMY_REPORT_DATA.analysis.findings.map((finding, index) => (
              <div key={index} className="finding-card">
                <div className="finding-header">
                  <h3>{finding.title}</h3>
                  <div className={`status-badge ${getStatusClass(finding.status)}`}>
                    {finding.status}
                  </div>
                </div>
                <p className="finding-description">{finding.description}</p>
                <div className="confidence-bar">
                  <span className="confidence-label">AI Confidence: {finding.confidence}%</span>
                  <div className="confidence-meter">
                    <div
                      className="confidence-fill"
                      style={{ width: `${finding.confidence}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Overall Assessment */}
        <section className="report-section assessment">
          <h2>Overall Assessment</h2>
          <div className="assessment-box">
            <p>{DUMMY_REPORT_DATA.analysis.overallAssessment}</p>
          </div>
        </section>

        {/* Recommendations */}
        <section className="report-section recommendations">
          <h2>Clinical Recommendations</h2>
          <ul className="recommendations-list">
            {DUMMY_REPORT_DATA.recommendations.map((rec, index) => (
              <li key={index}>{rec}</li>
            ))}
          </ul>
        </section>

        {/* Radiologist Information */}
        <section className="report-section radiologist-info">
          <h2>Radiologist Information</h2>
          <div className="info-grid">
            <div className="info-item">
              <span className="label">Radiologist:</span>
              <span className="value">{DUMMY_REPORT_DATA.radiologist.name}</span>
            </div>
            <div className="info-item">
              <span className="label">Specialty:</span>
              <span className="value">{DUMMY_REPORT_DATA.radiologist.specialty}</span>
            </div>
            <div className="info-item">
              <span className="label">License:</span>
              <span className="value">{DUMMY_REPORT_DATA.radiologist.license}</span>
            </div>
          </div>
        </section>

        {/* Raw Data Toggle */}
        <section className="report-section raw-data-section">
          <button
            className="btn btn-outline"
            onClick={() => setShowRawData(!showRawData)}
          >
            {showRawData ? "Hide Raw JSON Data" : "Show Raw JSON Data"}
          </button>

          {showRawData && (
            <pre className="raw-data">
              {JSON.stringify(DUMMY_REPORT_DATA, null, 2)}
            </pre>
          )}
        </section>

        {/* Footer */}
        <div className="report-footer">
          <p>This is a demonstration report with dummy data for testing purposes.</p>
          <p>Generated on: {new Date().toLocaleString()}</p>
        </div>
      </div>
    </div>
  );
}
