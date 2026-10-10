@echo off
echo ================================================
echo Tamil Sarcasm Detection - Dependency Installation
echo ================================================
echo.

echo Installing required packages...
echo.

pip install streamlit pandas numpy torch sentence-transformers scikit-learn matplotlib seaborn joblib tqdm

echo.
echo ================================================
echo Installation complete!
echo ================================================
echo.
echo To run the app, execute:
echo     streamlit run streamlit_app.py
echo.
pause

