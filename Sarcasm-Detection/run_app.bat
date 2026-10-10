@echo off
echo ================================================
echo Tamil Sarcasm Detection System
echo ================================================
echo.
echo Starting Streamlit app...
echo.
echo The app will open in your browser at http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

cd /d "%~dp0"
streamlit run streamlit_app.py
