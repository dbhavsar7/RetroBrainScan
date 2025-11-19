import { useState } from "react";
import UploadPage from "./UploadPage";
import ProcessingPage from "./ProcessingPage";
import DoctorReportPage from "./DoctorReportPage";
import "./App.css";

function App() {
  const [currentPage, setCurrentPage] = useState("upload");
  const [patientInfo, setPatientInfo] = useState(null);
  const [uploadPageKey, setUploadPageKey] = useState(0);
  const [autoFillAndSubmit, setAutoFillAndSubmit] = useState(false);

  const handleUploadComplete = (info, uploadedImages) => {
    setPatientInfo({...info, uploadedImages});
    setCurrentPage("processing");
  };

  const handleProcessingComplete = (analysisResults) => {
    setPatientInfo(prev => ({...prev, analysisResults}));
    setCurrentPage("report");
  };

  const handleBackToUpload = () => {
    setCurrentPage("upload");
    setPatientInfo(null);
  };

  return (
    <div className="d-flex flex-column min-vh-100">
      {currentPage !== "processing" && currentPage !== "report" && (
        <nav className="navbar navbar-expand-lg bg-white shadow">
          <div className="container-fluid">
            <a 
              className="navbar-brand fw-bold" 
              href="#" 
              onClick={(e) => {
                e.preventDefault();
                setCurrentPage("upload");
                setPatientInfo(null);
                setAutoFillAndSubmit(false);
                setUploadPageKey(prev => prev + 1);
              }}
            >
              <img 
                src="/RBS_Logo_T.png" 
                alt="RetroBrainScan" 
                className="navbar-logo"
              />
            </a>
            <button 
              className="navbar-toggler" 
              type="button" 
              data-bs-toggle="collapse" 
              data-bs-target="#navbarNav"
            >
              <span className="navbar-toggler-icon"></span>
            </button>
            <div className="collapse navbar-collapse" id="navbarNav">
              <ul className="navbar-nav ms-auto">
                <li className="nav-item">
                  <button 
                    className={`nav-link btn btn-link ${currentPage === "upload" ? "active" : ""}`}
                    onClick={() => {
                      setCurrentPage("upload");
                      setAutoFillAndSubmit(true);
                    }}
                  >
                    Upload Images
                  </button>
                </li>
              </ul>
            </div>
          </div>
        </nav>
      )}

      <main className="flex-grow-1 py-4">
        <div className={currentPage === "processing" || currentPage === "report" ? "" : "container-fluid"}>
          {currentPage === "upload" && <UploadPage key={uploadPageKey} onUploadComplete={handleUploadComplete} autoFillAndSubmit={autoFillAndSubmit} onAutoFillComplete={() => setAutoFillAndSubmit(false)} />}
          {currentPage === "processing" && <ProcessingPage onProcessingComplete={handleProcessingComplete} uploadedImages={patientInfo?.uploadedImages} patientInfo={patientInfo} />}
          {currentPage === "report" && <DoctorReportPage patientInfo={patientInfo} onBackClick={handleBackToUpload} />}
        </div>
      </main>
      {currentPage !== "processing" && currentPage !== "report" && (
        <footer className="bg-dark text-white text-center py-3 mt-auto">
          <p className="mb-0">
            <a href="https://hackrpi.com/" target="_blank" rel="noopener noreferrer" className="text-white text-decoration-none">HACKRPI 2025</a>
            {" X "}
            <a href="https://github.com/dbhavsar7/RetroBrainScan#" target="_blank" rel="noopener noreferrer" className="text-white text-decoration-none">RetroBrainScan</a>
          </p>
        </footer>
      )}
    </div>
  );
}

export default App;
