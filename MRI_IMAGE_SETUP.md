# MRI Brain Scan Image Setup

To display the MRI brain scan image in the report page and PDF:

## Frontend Setup

1. Place the `MRI_of_Human_Brain.jpg` file in the React frontend's public folder:
   ```
   React_Frontend/public/MRI_of_Human_Brain.jpg
   ```

   The image will be served at: `http://localhost:5173/MRI_of_Human_Brain.jpg`

## Backend Setup (PDF Generation)

1. Create the static/images directory in Flask_Backend:
   ```
   Flask_Backend/static/images/
   ```

2. Place the `MRI_of_Human_Brain.jpg` file there:
   ```
   Flask_Backend/static/images/MRI_of_Human_Brain.jpg
   ```

## How It Works

### Frontend (DoctorReportPage.jsx)
- Displays the MRI brain scan image above the report
- Image is served from the public folder
- Styled with a dark background and blue border matching the RetroBrainScan theme
- Responsive design that works on mobile and desktop

### Backend (Flask app.py)
- When generating a PDF, the app looks for the image at `static/images/MRI_of_Human_Brain.jpg`
- If found, it embeds the image into the PDF (4 inches x 4 inches)
- If not found, it logs a warning and continues without the image
- The image appears at the top of the generated PDF report

## Image Requirements

- **Format**: JPG, PNG, or GIF
- **Dimensions**: Recommended square or rectangular (ideally 512x512 or larger)
- **File Size**: Keep under 5MB for optimal PDF performance
- **Content**: MRI brain scan (axial view recommended)

## Testing

1. Upload an image through the upload page
2. Wait for processing to complete (5 seconds)
3. The report page will display the MRI brain scan image at the top
4. Click "Download PDF" to generate and download a PDF with the image included
