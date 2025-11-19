# Environment Variables Setup

## Google Gemini API Key Configuration

This project uses environment variables to securely store the Google Gemini API key.

### Setup Instructions

1. **Create a `.env` file** in the `Flask_Backend` directory (if it doesn't exist, the install script will create a template)

2. **Add your API key** to the `.env` file:
   ```
   GOOGLE_AI_API_KEY=your_actual_api_key_here
   GEMINI_MODEL=gemini-2.0-flash-exp
   ```

3. **Get your API key** from: https://makersuite.google.com/app/apikey

### Important Notes

- The `.env` file is already in `.gitignore` and will NOT be committed to version control
- Never commit your actual API key to the repository
- The `.env.example` file shows the required format without exposing secrets
- If the API key is not set, Gemini features will be disabled with a warning message

### Model Options

You can change the `GEMINI_MODEL` variable to use different models:
- `gemini-2.0-flash-exp` (default, recommended)
- `gemini-1.5-pro-latest`
- `gemini-1.5-pro`

### Verification

After setting up the `.env` file, restart the Flask server. You should see:
- No warning messages about missing API key
- Gemini features working correctly

If you see: `"Warning: GOOGLE_AI_API_KEY environment variable not set"`, check that:
1. The `.env` file exists in `Flask_Backend/` directory
2. The file contains `GOOGLE_AI_API_KEY=your_key_here`
3. There are no extra spaces or quotes around the key

