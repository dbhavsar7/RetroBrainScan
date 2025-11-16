import { useState } from "react";
import UploadPage from "./UploadPage";
import ProcessingPage from "./ProcessingPage";
import DoctorReportPage from "./DoctorReportPage";
import MessageSender from "./MessageSender";
import "./App.css";

function App() {
  const [currentPage, setCurrentPage] = useState("upload");

  const handleUploadComplete = () => {
    // After upload, go to processing page
    setCurrentPage("processing");
  };

  const handleProcessingComplete = () => {
    // After processing (5 seconds), go to report page
    setCurrentPage("report");
  };

  const handleBackToUpload = () => {
    setCurrentPage("upload");
  };

  return (
    <div className="d-flex flex-column min-vh-100">
      {/* Navigation - Hidden on processing and report pages */}
      {currentPage !== "processing" && currentPage !== "report" && (
        <nav className="navbar navbar-expand-lg navbar-dark bg-dark shadow">
          <div className="container-fluid">
            <a 
              className="navbar-brand fw-bold" 
              href="#" 
              onClick={(e) => {
                e.preventDefault();
                setCurrentPage("upload");
              }}
            >
              🧠 RetroBrainScan
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
                    onClick={() => setCurrentPage("upload")}
                  >
                    Upload Images
                  </button>
                </li>
                <li className="nav-item">
                  <button 
                    className={`nav-link btn btn-link ${currentPage === "test" ? "active" : ""}`}
                    onClick={() => setCurrentPage("test")}
                  >
                    Test API
                  </button>
                </li>
              </ul>
            </div>
          </div>
        </nav>
      )}

      {/* Main Content */}
      <main className="flex-grow-1 py-4">
        <div className={currentPage === "processing" || currentPage === "report" ? "" : "container"}>
          {currentPage === "upload" && <UploadPage onUploadComplete={handleUploadComplete} />}
          {currentPage === "processing" && <ProcessingPage onProcessingComplete={handleProcessingComplete} />}
          {currentPage === "report" && <DoctorReportPage onBackClick={handleBackToUpload} />}
          {currentPage === "test" && <MessageSender />}
        </div>
      </main>

      {/* Footer - Hidden on processing and report pages */}
      {currentPage !== "processing" && currentPage !== "report" && (
        <footer className="bg-dark text-white text-center py-3 mt-auto">
          <p className="mb-0">&copy; 2025 RetroBrainScan. All rights reserved.</p>
        </footer>
      )}
    </div>
  );
}

export default App;
