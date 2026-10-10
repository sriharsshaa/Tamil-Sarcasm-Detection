# Tamil Sarcasm Detection System

A comprehensive system for detecting sarcasm in Tamil news headlines using state-of-the-art NLP techniques.

## Features

🔍 **Interactive Streamlit Dashboard** with:
- **Data Exploration**: Visualize dataset statistics and class distribution
- **Preprocessing Steps**: See all intermediate preprocessing transformations
- **Model Training Progress**: Track training metrics and learning curves
- **Accuracy Metrics**: Detailed performance evaluation with ROC curves
- **Live Prediction**: Test the model with custom Tamil headlines

## Technology Stack

- **Embeddings**: LaBSE / Multilingual MiniLM for sentence embeddings
- **Dependency Parsing**: CoNLL-U format with edge-aware features
- **Commonsense Knowledge**: COMET for contextual understanding
- **Model**: Residual MLP neural network with PyTorch
- **UI**: Streamlit for interactive visualization

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Required Files

Keep these files in the project directory alongside `streamlit_app.py`:
- `dataset_comet.csv` or `Headlines_Tamil_Final_Merged.csv` - Dataset (kept local; not included in Git)
- `best_model_concat.pth` - Trained PyTorch model
- `scaler.pkl` - StandardScaler for feature normalization
- `training_history.json` - Training metrics history
- `mlp_labse_tamil_joblib.pkl` - Alternative sklearn model (optional)

Dataset CSVs are excluded by `.gitignore` and must be provided locally. GitHub
does not support making individual files private inside a public repository.

## Running the App

### Option 1: Using PowerShell (Windows)

```powershell
cd path\to\sarcasmDetection
streamlit run streamlit_app.py
```

### Option 2: Using Command Prompt

```cmd
cd path\to\sarcasmDetection
streamlit run streamlit_app.py
```

The app will automatically open in your default browser at `http://localhost:8501`

## Dashboard Pages

### 1. **Home**
- Quick overview of the system
- Dataset statistics at a glance
- Technology stack information

### 2. **Data Exploration**
- Dataset shape and columns
- Class distribution (Sarcastic vs Non-Sarcastic)
- Sample Tamil headlines
- Interactive visualizations

### 3. **Preprocessing Steps**
Visualize all preprocessing stages:
- **Step 1**: Data loading and cleaning
- **Step 2**: Tamil text preprocessing
- **Step 3**: Sentence embeddings generation
- **Step 4**: Dependency parsing (CoNLL-U format)
- **Step 5**: COMET commonsense features
- **Step 6**: Feature concatenation (1152 dimensions)
- **Step 7**: Standardization

### 4. **Model Training**
- Residual MLP architecture visualization
- Training configuration details
- Accuracy and loss curves over epochs
- Learning rate schedule
- Training statistics

### 5. **Accuracy Metrics**
- Final performance summary
- Train/Validation/Test accuracy
- ROC AUC scores
- Overfitting analysis
- Training stability metrics
- Epoch-by-epoch detailed metrics table

### 6. **Live Prediction**
- Enter custom Tamil headlines
- Real-time sarcasm prediction
- Confidence scores
- Probability distribution visualization
- Pre-loaded examples

## Model Architecture

```
ResidualMLP(
  Input Layer: 1152 → 512
  ResidualBlock(512)
  Linear: 512 → 256
  ResidualBlock(256)
  Linear: 256 → 128
  ResidualBlock(128)
  Output Layer: 128 → 2 (classes)
)
```

### Training Configuration
- **Optimizer**: AdamW
- **Learning Rate**: 1e-3
- **Scheduler**: CosineAnnealingWarmRestarts
- **Batch Size**: 64
- **Dropout**: 0.35
- **Weight Decay**: 1e-4
- **Early Stopping**: Patience of 8 epochs

## Feature Pipeline

The system combines three types of features:

1. **Tamil Headline Embeddings** (384 dims)
   - Using paraphrase-multilingual-MiniLM-L12-v2
   - L2 normalized

2. **Dependency Parse Features** (384 dims)
   - Parsed from CoNLL-U format
   - Edge-aware representations: `word[relation->head]`
   - Captures syntactic structure

3. **COMET Commonsense Features** (384 dims)
   - xIntent, xEffect, xReact, oReact, oWant
   - Provides contextual understanding

**Total**: 1152-dimensional feature vector

## Dataset Format

The system expects CSV files with these columns:

### Minimal Format (Headlines_Tamil_Final_Merged.csv):
- `headline_tamil`: Tamil text
- `is_sarcastic`: Binary label (0/1)

### Full Format (dataset_comet.csv):
- `headline_tamil`: Tamil text
- `is_sarcastic`: Binary label
- `conllu`: Dependency parse in CoNLL-U format
- `comet_xIntent`: COMET intent inference
- `comet_xEffect`: COMET effect inference
- `comet_xReact`: COMET subject reaction
- `comet_oReact`: COMET object reaction
- `comet_oWant`: COMET object desire

## Troubleshooting

### Model Not Loading
- Ensure `best_model_concat.pth` exists in the project directory
- Check that `scaler.pkl` is present
- Verify PyTorch installation: `pip install torch`

### Training History Not Found
- The app will display a warning but other pages will still work
- Run the training script to generate `training_history.json`

### Embedder Loading Issues
- Ensure sentence-transformers is installed: `pip install sentence-transformers`
- First run will download the model (~400MB)
- Requires internet connection for first-time download

### Memory Issues
- Reduce batch size in the code
- Use CPU instead of GPU if OOM errors occur
- Close other applications to free up RAM

## Performance Metrics

Based on training history:
- **Training Accuracy**: ~99%
- **Validation Accuracy**: ~98%
- **Test Accuracy**: ~97%
- **ROC AUC**: ~0.99

The model shows excellent generalization with minimal overfitting.

## Project Structure

```
sarcasmDetection\
├── streamlit_app.py                    # Main Streamlit application
├── copy_of_alternate_sarcasm.py        # Training scripts
├── dataset_comet.csv                   # Local-only; excluded from Git
├── dataset_depparsed.csv               # Local-only; excluded from Git
├── Headlines_Tamil_Final_Merged.csv    # Local-only; excluded from Git
├── best_model_concat.pth               # Trained PyTorch model weights
├── scaler.pkl                          # Feature scaler
├── training_history.json               # Training metrics log
├── mlp_labse_tamil_joblib.pkl          # Alternative sklearn model
└── README.md                           # This file
```

## Future Enhancements

- [ ] Add confusion matrix visualization
- [ ] Export predictions to CSV
- [ ] Batch prediction support
- [ ] Model comparison dashboard
- [ ] Real-time COMET feature generation
- [ ] Dependency parsing integration
- [ ] Multi-model ensemble predictions

## Citation

If you use this system, please cite:

```bibtex
@software{tamil_sarcasm_detection,
  title = {Tamil Sarcasm Detection System},
  author = {Your Name},
  year = {2025},
  description = {Deep learning system for sarcasm detection in Tamil news headlines}
}
```

## License

MIT License - Feel free to use and modify for your projects.

## Contact

For questions or issues, please open an issue in the repository.

---

**Built with ❤️ using Streamlit, PyTorch, and Transformers**

## Optional ngrok tunnel

The archived notebook script uses ngrok for optional public tunnel access. Set
the `NGROK_AUTHTOKEN` environment variable before running those notebook cells.
In PowerShell, set it for the current terminal session with:

```powershell
$env:NGROK_AUTHTOKEN = "your ngrok token"
```

Do not replace this with a real token in source code or commit the token to the
repository. The Streamlit app itself does not require ngrok.
