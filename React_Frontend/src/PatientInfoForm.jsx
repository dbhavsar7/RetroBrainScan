import { useState, useEffect } from "react";
import "./PatientInfoForm.css";

export default function PatientInfoForm({ onSubmit, onCancel, isLoading, autoFillAndSubmit, onAutoFillComplete }) {
  const [patientData, setPatientData] = useState({
    patient: {
      name: "",
      age: "",
      gender: "Male",
      mrn: "",
      dateOfBirth: "",
      hospital: "General Medical Center",
    },
    doctor: {
      name: "",
      specialty: "Neuroradiology",
      license: "",
    },
  });

  const [errors, setErrors] = useState({});

  const handlePatientChange = (field, value) => {
    setPatientData((prev) => ({
      ...prev,
      patient: { ...prev.patient, [field]: value },
    }));
    // Clear error for this field
    if (errors[`patient_${field}`]) {
      setErrors((prev) => ({
        ...prev,
        [`patient_${field}`]: "",
      }));
    }
  };

  const handleDoctorChange = (field, value) => {
    setPatientData((prev) => ({
      ...prev,
      doctor: { ...prev.doctor, [field]: value },
    }));
    // Clear error for this field
    if (errors[`doctor_${field}`]) {
      setErrors((prev) => ({
        ...prev,
        [`doctor_${field}`]: "",
      }));
    }
  };

  const validateForm = () => {
    const newErrors = {};

    if (!patientData.patient.name.trim()) {
      newErrors.patient_name = "Patient name is required";
    }
    if (!patientData.patient.age || patientData.patient.age <= 0) {
      newErrors.patient_age = "Valid age is required";
    }
    if (!patientData.patient.mrn.trim()) {
      newErrors.patient_mrn = "MRN is required";
    }
    if (!patientData.patient.dateOfBirth) {
      newErrors.patient_dateOfBirth = "Date of birth is required";
    }
    if (!patientData.doctor.name.trim()) {
      newErrors.doctor_name = "Doctor name is required";
    }
    if (!patientData.doctor.license.trim()) {
      newErrors.doctor_license = "License number is required";
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (validateForm()) {
      onSubmit(patientData);
    }
  };

  const handleAutofill = () => {
    setPatientData({
      patient: {
        name: "Cake Jlouse",
        age: "30",
        gender: "Male",
        mrn: "MRN-2025-001116",
        dateOfBirth: "1995-11-06", // Format: YYYY-MM-DD for date input
        hospital: "General Medical Center",
      },
      doctor: {
        name: "Dr. Brew Dhavsar",
        specialty: "Neuroradiology",
        license: "#NR-111625",
      },
    });
    // Clear any existing errors
    setErrors({});
  };

  // Auto-fill and submit when triggered from navigation
  useEffect(() => {
    if (autoFillAndSubmit) {
      // Auto-fill the form
      handleAutofill();
      
      // Wait a brief moment for state to update, then submit
      const timer = setTimeout(() => {
        const autofilledData = {
          patient: {
            name: "Cake Jlouse",
            age: "30",
            gender: "Male",
            mrn: "MRN-2025-001116",
            dateOfBirth: "1995-11-06",
            hospital: "General Medical Center",
          },
          doctor: {
            name: "Dr. Brew Dhavsar",
            specialty: "Neuroradiology",
            license: "#NR-111625",
          },
        };
        
        // Submit the form
        onSubmit(autofilledData);
        
        // Notify parent that auto-fill is complete
        if (onAutoFillComplete) {
          onAutoFillComplete();
        }
      }, 100);
      
      return () => clearTimeout(timer);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoFillAndSubmit]);

  return (
    <div className="patient-info-form-container">
      <div className="form-card">
        <h2 className="form-title">👤 Patient & Doctor Information</h2>
        <p className="form-subtitle">
          Please fill in the patient and doctor details for the scan report
        </p>

        <form onSubmit={handleSubmit} className="patient-form">
          {/* Patient Information Section */}
          <div className="form-section">
            <div className="section-title-wrapper">
              <h3 className="section-title">Patient Information</h3>
              <button
                type="button"
                className="autofill-btn"
                onClick={handleAutofill}
                title="Autofill sample patient information"
              >
                <span className="autofill-icon">⚡</span>
                <span className="autofill-text">Autofill</span>
              </button>
            </div>

            <div className="form-group">
              <label htmlFor="patient-name">Patient Name *</label>
              <input
                type="text"
                id="patient-name"
                placeholder="Enter full name"
                value={patientData.patient.name}
                onChange={(e) => handlePatientChange("name", e.target.value)}
                className={errors.patient_name ? "error" : ""}
              />
              {errors.patient_name && (
                <span className="error-text">{errors.patient_name}</span>
              )}
            </div>

            <div className="form-row">
              <div className="form-group">
                <label htmlFor="patient-age">Age *</label>
                <input
                  type="number"
                  id="patient-age"
                  placeholder="Age"
                  min="1"
                  max="120"
                  value={patientData.patient.age}
                  onChange={(e) => handlePatientChange("age", e.target.value)}
                  className={errors.patient_age ? "error" : ""}
                />
                {errors.patient_age && (
                  <span className="error-text">{errors.patient_age}</span>
                )}
              </div>

              <div className="form-group">
                <label htmlFor="patient-gender">Gender</label>
                <select
                  id="patient-gender"
                  value={patientData.patient.gender}
                  onChange={(e) =>
                    handlePatientChange("gender", e.target.value)
                  }
                >
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                  <option value="Other">Other</option>
                </select>
              </div>
            </div>

            <div className="form-group">
              <label htmlFor="patient-mrn">Medical Record Number (MRN) *</label>
              <input
                type="text"
                id="patient-mrn"
                placeholder="e.g., MRN-2025-001234"
                value={patientData.patient.mrn}
                onChange={(e) => handlePatientChange("mrn", e.target.value)}
                className={errors.patient_mrn ? "error" : ""}
              />
              {errors.patient_mrn && (
                <span className="error-text">{errors.patient_mrn}</span>
              )}
            </div>

            <div className="form-row">
              <div className="form-group">
                <label htmlFor="patient-dob">Date of Birth *</label>
                <input
                  type="date"
                  id="patient-dob"
                  value={patientData.patient.dateOfBirth}
                  onChange={(e) =>
                    handlePatientChange("dateOfBirth", e.target.value)
                  }
                  className={errors.patient_dateOfBirth ? "error" : ""}
                />
                {errors.patient_dateOfBirth && (
                  <span className="error-text">
                    {errors.patient_dateOfBirth}
                  </span>
                )}
              </div>

              <div className="form-group">
                <label htmlFor="patient-hospital">Hospital</label>
                <input
                  type="text"
                  id="patient-hospital"
                  placeholder="Hospital name"
                  value={patientData.patient.hospital}
                  onChange={(e) =>
                    handlePatientChange("hospital", e.target.value)
                  }
                />
              </div>
            </div>
          </div>

          {/* Doctor Information Section */}
          <div className="form-section">
            <h3 className="section-title">Radiologist Information</h3>

            <div className="form-group">
              <label htmlFor="doctor-name">Radiologist Name *</label>
              <input
                type="text"
                id="doctor-name"
                placeholder="Enter doctor's full name"
                value={patientData.doctor.name}
                onChange={(e) => handleDoctorChange("name", e.target.value)}
                className={errors.doctor_name ? "error" : ""}
              />
              {errors.doctor_name && (
                <span className="error-text">{errors.doctor_name}</span>
              )}
            </div>

            <div className="form-row">
              <div className="form-group">
                <label htmlFor="doctor-specialty">Specialty</label>
                <select
                  id="doctor-specialty"
                  value={patientData.doctor.specialty}
                  onChange={(e) =>
                    handleDoctorChange("specialty", e.target.value)
                  }
                >
                  <option value="Neuroradiology">Neuroradiology</option>
                  <option value="General Radiology">General Radiology</option>
                  <option value="Interventional Radiology">
                    Interventional Radiology
                  </option>
                  <option value="Other">Other</option>
                </select>
              </div>

              <div className="form-group">
                <label htmlFor="doctor-license">License Number *</label>
                <input
                  type="text"
                  id="doctor-license"
                  placeholder="e.g., License #NR-45628"
                  value={patientData.doctor.license}
                  onChange={(e) =>
                    handleDoctorChange("license", e.target.value)
                  }
                  className={errors.doctor_license ? "error" : ""}
                />
                {errors.doctor_license && (
                  <span className="error-text">{errors.doctor_license}</span>
                )}
              </div>
            </div>
          </div>

          {/* Form Actions */}
          <div className="form-actions">
            <button
              type="submit"
              className="btn btn-primary"
              disabled={isLoading}
            >
              {isLoading ? "Loading..." : "Continue to Upload"}
            </button>
          </div>

          <p className="form-note">* Required fields</p>
        </form>
      </div>
    </div>
  );
}
