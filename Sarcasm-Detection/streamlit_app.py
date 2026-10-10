# -*- coding: utf-8 -*-
"""
Tamil Sarcasm Detection - Streamlit Dashboard
Displays preprocessing steps, model training, and accuracy metrics
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Set page config
st.set_page_config(page_title="Tamil Sarcasm Detection", page_icon="🔍", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2ca02c;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Paths
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_PATH, "dataset_comet.csv")
HEADLINES_CSV = os.path.join(BASE_PATH, "Headlines_Tamil_Final_Merged.csv")
HISTORY_PATH = os.path.join(BASE_PATH, "training_history.json")
MODEL_PATH = os.path.join(BASE_PATH, "best_model_concat.pth")
SCALER_PATH = os.path.join(BASE_PATH, "scaler.pkl")
MLP_JOBLIB_PATH = os.path.join(BASE_PATH, "mlp_labse_tamil_joblib.pkl")

# Define the model architecture (same as training)
class ResidualBlock(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.3):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.proj = None
        if hidden_dim != dim:
            self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.bn2(out)
        res = self.proj(x) if self.proj is not None else x
        return self.act(out + res)

class ResidualMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], dropout=0.35):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, hidden_dims[0])
        self.input_bn = nn.BatchNorm1d(hidden_dims[0])
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

        blocks = []
        for i, dim in enumerate(hidden_dims):
            blocks.append(ResidualBlock(dim, hidden_dim=max(dim//2, 64), dropout=dropout))
            if i + 1 < len(hidden_dims):
                next_dim = hidden_dims[i+1]
                blocks.append(nn.Linear(dim, next_dim))
                blocks.append(nn.BatchNorm1d(next_dim))
                blocks.append(nn.ReLU(inplace=True))
                blocks.append(nn.Dropout(dropout))
        self.blocks = nn.Sequential(*blocks)
        final_dim = hidden_dims[-1]
        self.head = nn.Sequential(
            nn.Linear(final_dim, max(final_dim//2, 64)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(max(final_dim//2, 64), 2)
        )

    def forward(self, x):
        x = self.input_fc(x)
        x = self.input_bn(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.blocks(x)
        x = self.head(x)
        return x

# Title
st.markdown('<h1 class="main-header">🔍 Tamil Sarcasm Detection System</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Exploration", "Preprocessing Steps", "Model Training", "Accuracy Metrics", "Live Prediction"])

# Cache functions
@st.cache_data
def load_data():
    """Load dataset"""
    try:
        if os.path.exists(CSV_PATH):
            df = pd.read_csv(CSV_PATH, encoding='utf-8', engine='python')
        else:
            df = pd.read_csv(HEADLINES_CSV, encoding='utf-8', engine='python')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def load_training_history():
    """Load training history"""
    try:
        with open(HISTORY_PATH, 'r') as f:
            history = json.load(f)
        return history
    except Exception as e:
        st.warning(f"Training history not found: {e}")
        return None

@st.cache_resource
def load_model():
    """Load trained model"""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Try loading the PyTorch model first
        if os.path.exists(MODEL_PATH):
            # Dummy input_dim - will be set from checkpoint
            checkpoint = torch.load(MODEL_PATH, map_location=device)
            
            # Infer input_dim from state dict
            input_dim = checkpoint['model_state_dict']['input_fc.weight'].shape[1]
            
            model = ResidualMLP(input_dim, hidden_dims=[512, 256, 128], dropout=0.35).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            scaler = joblib.load(SCALER_PATH)
            return model, scaler, device, "ResidualMLP"
        
        # Fallback to joblib MLP
        elif os.path.exists(MLP_JOBLIB_PATH):
            mlp = joblib.load(MLP_JOBLIB_PATH)
            return mlp, None, device, "MLPClassifier"
        
        return None, None, None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None

@st.cache_resource
def load_embedder():
    """Load sentence embedder"""
    try:
        embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        return embedder
    except Exception as e:
        st.error(f"Error loading embedder: {e}")
        return None

def parse_conllu_edges(conllu_str):
    """Parse CoNLL-U string and return edge-aware token string"""
    try:
        if not isinstance(conllu_str, str) or conllu_str.strip() == "":
            return ""
        lines = [l for l in conllu_str.splitlines() if l.strip() and not l.strip().startswith("#")]
        token_dict = {}
        head_dict = {}
        for ln in lines:
            parts = ln.split("\t")
            if len(parts) >= 8:
                idx = parts[0]
                if not idx.isdigit():
                    continue
                idx = int(idx)
                word = parts[1]
                head = int(parts[6])
                rel = parts[7]
                token_dict[idx] = word
                head_dict[idx] = (head, rel)
        edge_texts = []
        for idx, word in token_dict.items():
            head, rel = head_dict[idx]
            if head == 0:
                edge_texts.append(f"{word}[root]")
            else:
                head_word = token_dict.get(head, "ROOT")
                edge_texts.append(f"{word}[{rel}->{head_word}]")
        return " ".join(edge_texts)
    except Exception:
        return ""

def collect_comet_text(row):
    """Collect COMET fields into text"""
    fields = [
        row.get("comet_xIntent", ""),
        row.get("comet_xEffect", ""),
        row.get("comet_xReact", ""),
        row.get("comet_oReact", ""),
        row.get("comet_oWant", "")
    ]
    texts = []
    for f in fields:
        if f is None:
            continue
        s = str(f).strip()
        if s == "" or s.lower() == "none":
            continue
        texts.append(s)
    if len(texts) == 0:
        return ""
    return " . ".join(texts)

# Pages
if page == "Home":
    st.markdown("## Welcome to Tamil Sarcasm Detection System")
    
    st.markdown("""
    This system detects sarcasm in Tamil news headlines using advanced NLP techniques.
    
    ### Features:
    - 📊 **Data Exploration**: View and analyze the dataset
    - 🔧 **Preprocessing Steps**: See how data is transformed
    - 🤖 **Model Training**: Track model training progress
    - 📈 **Accuracy Metrics**: Detailed performance evaluation
    - 🎯 **Live Prediction**: Test the model with custom inputs
    
    ### Technology Stack:
    - **LaBSE** / **Multilingual MiniLM** for embeddings
    - **Dependency Parsing** with CoNLL-U format
    - **COMET** for commonsense knowledge
    - **Residual MLP** neural network
    - **PyTorch** for deep learning
    """)
    
    # Display quick stats
    df = load_data()
    if df is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Samples", len(df))
        with col2:
            st.metric("Features", len(df.columns))
        
        # Commented out to avoid ValueError
        # with col2:
        #     if 'is_sarcastic' in df.columns:
        #         sarcastic_count = df['is_sarcastic'].sum()
        #         st.metric("Sarcastic", int(sarcastic_count) if pd.notna(sarcastic_count) else 0)
        # with col3:
        #     if 'is_sarcastic' in df.columns:
        #         non_sarcastic_count = (df['is_sarcastic'] == 0).sum()
        #         st.metric("Non-Sarcastic", int(non_sarcastic_count) if pd.notna(non_sarcastic_count) else 0)

elif page == "Data Exploration":
    st.markdown('<div class="section-header">📊 Data Exploration</div>', unsafe_allow_html=True)
    
    df = load_data()
    if df is not None:
        # Show basic info
        st.markdown("### Dataset Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            st.write(f"**Columns:** {', '.join(df.columns.tolist()[:5])}...")
        
        # Class distribution
        if 'is_sarcastic' in df.columns:
            st.markdown("### Class Distribution")
            col1, col2 = st.columns([1, 2])
            
            with col1:
                class_counts = df['is_sarcastic'].value_counts()
                st.dataframe(pd.DataFrame({
                    'Class': ['Non-Sarcastic', 'Sarcastic'],
                    'Count': [class_counts.get(0, 0), class_counts.get(1, 0)],
                    'Percentage': [
                        f"{class_counts.get(0, 0) / len(df) * 100:.2f}%",
                        f"{class_counts.get(1, 0) / len(df) * 100:.2f}%"
                    ]
                }))
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.countplot(data=df, x='is_sarcastic', palette=['#2ca02c', '#d62728'], ax=ax)
                ax.set_xlabel('Class')
                ax.set_ylabel('Count')
                ax.set_title('Sarcasm Distribution')
                ax.set_xticklabels(['Non-Sarcastic', 'Sarcastic'])
                st.pyplot(fig)
        
        # Sample data
        st.markdown("### Sample Headlines")
        if 'headline_tamil' in df.columns:
            sample_df = df[['headline_tamil', 'is_sarcastic']].head(10) if 'is_sarcastic' in df.columns else df[['headline_tamil']].head(10)
            st.dataframe(sample_df, use_container_width=True)
        else:
            st.dataframe(df.head(10), use_container_width=True)

elif page == "Preprocessing Steps":
    st.markdown('<div class="section-header">🔧 Preprocessing Pipeline</div>', unsafe_allow_html=True)
    
    df = load_data()
    
    if df is not None:
        st.markdown("### Step 1: Data Loading & Cleaning")
        st.info(f"✅ Loaded {len(df)} samples from CSV")
        
        if 'headline_tamil' in df.columns:
            # Show text preprocessing
            st.markdown("### Step 2: Text Preprocessing")
            st.write("**Tamil Text Example:**")
            sample_text = df['headline_tamil'].iloc[0] if len(df) > 0 else "N/A"
            st.code(sample_text, language='text')
            
            # Show embedding generation
            st.markdown("### Step 3: Sentence Embeddings")
            with st.spinner("Loading embedder..."):
                embedder = load_embedder()
                if embedder is not None:
                    st.success("✅ Loaded paraphrase-multilingual-MiniLM-L12-v2")
                    
                    # Demonstrate embedding
                    if st.button("Generate Sample Embedding"):
                        with st.spinner("Generating embedding..."):
                            emb = embedder.encode([sample_text], convert_to_numpy=True)
                            st.write(f"**Embedding Shape:** {emb.shape}")
                            st.write(f"**Embedding Vector (first 10 dims):** {emb[0][:10]}")
        
        # Show dependency parsing
        if 'conllu' in df.columns:
            st.markdown("### Step 4: Dependency Parsing (CoNLL-U)")
            st.write("**Raw CoNLL-U Format:**")
            sample_conllu = df['conllu'].iloc[0] if len(df) > 0 else "N/A"
            st.text(sample_conllu[:300] + "..." if len(str(sample_conllu)) > 300 else sample_conllu)
            
            st.write("**Parsed Edge-Aware Tokens:**")
            parsed = parse_conllu_edges(sample_conllu)
            st.code(parsed[:200] + "..." if len(parsed) > 200 else parsed, language='text')
        
        # Show COMET features
        if all(col in df.columns for col in ['comet_xIntent', 'comet_xEffect', 'comet_xReact']):
            st.markdown("### Step 5: COMET Commonsense Features")
            sample_row = df.iloc[0]
            comet_text = collect_comet_text(sample_row)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Individual COMET Fields:**")
                for field in ['comet_xIntent', 'comet_xEffect', 'comet_xReact', 'comet_oReact', 'comet_oWant']:
                    if field in sample_row:
                        st.write(f"- **{field}:** {sample_row[field]}")
            
            with col2:
                st.write("**Combined COMET Text:**")
                st.code(comet_text[:200] + "..." if len(comet_text) > 200 else comet_text, language='text')
        
        # Show feature concatenation
        st.markdown("### Step 6: Feature Concatenation")
        st.write("""
        Final features are created by concatenating:
        1. **Tamil Headline Embeddings** (384 dims)
        2. **Dependency Parse Embeddings** (384 dims)
        3. **COMET Embeddings** (384 dims)
        
        **Total Feature Dimension:** 1152
        """)
        
        # Show normalization
        st.markdown("### Step 7: Standardization")
        st.write("Features are standardized using StandardScaler (mean=0, std=1)")

elif page == "Model Training":
    st.markdown('<div class="section-header">🤖 Model Training Progress</div>', unsafe_allow_html=True)
    
    history = load_training_history()
    
    if history is not None:
        # Model architecture
        st.markdown("### Model Architecture: Residual MLP")
        st.code("""
ResidualMLP(
  Input Layer: 1152 -> 512
  ResidualBlock(512)
  Linear: 512 -> 256
  ResidualBlock(256)
  Linear: 256 -> 128
  ResidualBlock(128)
  Output Layer: 128 -> 2 (classes)
)
        """, language='text')
        
        # Training configuration
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Training Configuration")
            st.write("""
            - **Optimizer:** AdamW
            - **Learning Rate:** 1e-3
            - **Scheduler:** CosineAnnealingWarmRestarts
            - **Batch Size:** 64
            - **Dropout:** 0.35
            - **Weight Decay:** 1e-4
            """)
        
        with col2:
            st.markdown("### Training Statistics")
            st.write(f"- **Total Epochs:** {len(history['train_loss'])}")
            st.write(f"- **Best Val Accuracy:** {max(history['val_acc'])*100:.2f}%")
            st.write(f"- **Final Train Accuracy:** {history['train_acc'][-1]*100:.2f}%")
            st.write(f"- **Final Val Accuracy:** {history['val_acc'][-1]*100:.2f}%")
        
        # Training curves
        st.markdown("### Training Progress")
        
        # Accuracy plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(history['train_acc']) + 1)
        ax1.plot(epochs, [x*100 for x in history['train_acc']], label='Train Accuracy', linewidth=2)
        ax1.plot(epochs, [x*100 for x in history['val_acc']], label='Validation Accuracy', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Accuracy over Epochs')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss plot
        ax2.plot(epochs, history['train_loss'], label='Train Loss', linewidth=2)
        ax2.plot(epochs, history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Loss over Epochs')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # Learning rate schedule
        if 'lr' in history:
            st.markdown("### Learning Rate Schedule")
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(epochs, history['lr'], linewidth=2, color='purple')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate over Epochs')
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            st.pyplot(fig)
    else:
        st.warning("Training history not available. Please run the training script first.")

elif page == "Accuracy Metrics":
    st.markdown('<div class="section-header">📈 Model Performance Metrics</div>', unsafe_allow_html=True)
    
    history = load_training_history()
    
    if history is not None:
        # Summary metrics
        st.markdown("### Final Performance Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Train Accuracy", f"{history['train_acc'][-1]*100:.2f}%")
        with col2:
            st.metric("Validation Accuracy", f"{history['val_acc'][-1]*100:.2f}%")
        with col3:
            best_val = max(history['val_acc'])
            st.metric("Best Val Accuracy", f"{best_val*100:.2f}%")
        with col4:
            if 'val_roc_auc' in history:
                st.metric("Val ROC AUC", f"{history['val_roc_auc'][-1]:.4f}")
        
        # ROC AUC plot
        if 'val_roc_auc' in history and 'train_roc_auc' in history:
            st.markdown("### ROC AUC Score over Epochs")
            fig, ax = plt.subplots(figsize=(12, 5))
            epochs = range(1, len(history['train_roc_auc']) + 1)
            ax.plot(epochs, history['train_roc_auc'], label='Train ROC AUC', linewidth=2)
            ax.plot(epochs, history['val_roc_auc'], label='Validation ROC AUC', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('ROC AUC Score')
            ax.set_title('ROC AUC over Epochs')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        # Detailed metrics table
        st.markdown("### Epoch-by-Epoch Metrics")
        metrics_df = pd.DataFrame({
            'Epoch': range(1, len(history['train_acc']) + 1),
            'Train Acc': [f"{x*100:.2f}%" for x in history['train_acc']],
            'Val Acc': [f"{x*100:.2f}%" for x in history['val_acc']],
            'Train Loss': [f"{x:.4f}" for x in history['train_loss']],
            'Val Loss': [f"{x:.4f}" for x in history['val_loss']],
        })
        st.dataframe(metrics_df, use_container_width=True, height=400)
    else:
        st.warning("Training metrics not available. Please run the training script first.")

elif page == "Live Prediction":
    st.markdown('<div class="section-header">🎯 Live Sarcasm Prediction</div>', unsafe_allow_html=True)
    
    model, scaler, device, model_type = load_model()
    embedder = load_embedder()
    
    if model is not None and embedder is not None:
        st.markdown("### Enter Tamil Headline")
        
        # Sample examples
        examples = [
            "ஏப்ரல் மாதத்தில் ஒரு பாகிஸ்தான் நகரம் 122.4 டிகிரியைத் தாக்கியது, அநேகமாக உலக சாதனை படைத்திருக்கலாம்",
            "யு.எஸ்.மிட்நைட் ரெய்டில் முக்கிய ஈராக் தளங்களை வெளியே எடுக்கிறது",
        ]
        
        col1, col2 = st.columns([3, 1])
        with col1:
            text_input = st.text_area("Tamil Text:", height=100, placeholder="Enter Tamil headline here...")
        with col2:
            st.markdown("**Examples:**")
            for i, ex in enumerate(examples):
                if st.button(f"Example {i+1}", key=f"ex{i}"):
                    text_input = ex
                    st.rerun()
        
        # Option to show intermediate steps
        show_steps = st.checkbox("Show intermediate processing steps", value=True)
        
        if st.button("🔍 Predict", type="primary"):
            if text_input.strip():
                with st.spinner("Processing..."):
                    try:
                        # Show preprocessing steps
                        if show_steps:
                            st.markdown("---")
                            st.markdown("## 🔄 Preprocessing Pipeline")
                        
                        # Step 1: Raw Input
                        if show_steps:
                            with st.expander("📝 Step 1: Raw Input Text", expanded=True):
                                st.code(text_input, language='text')
                                st.info(f"Text length: {len(text_input)} characters")
                        
                        # Step 2: Generate headline embedding
                        if show_steps:
                            st.markdown("### ⚙️ Step 2: Generating Tamil Headline Embeddings")
                            with st.spinner("Encoding text with multilingual embedder..."):
                                headline_emb = embedder.encode([text_input], convert_to_numpy=True)
                            
                            with st.expander("🔢 Headline Embedding Details", expanded=False):
                                st.write(f"**Shape:** {headline_emb.shape}")
                                st.write(f"**Dimensions:** {headline_emb.shape[1]}")
                                st.write(f"**First 20 values:**")
                                st.code(str(headline_emb[0][:20]), language='python')
                                st.write(f"**Statistics:**")
                                col1, col2, col3, col4 = st.columns(4)
                                col1.metric("Mean", f"{headline_emb.mean():.4f}")
                                col2.metric("Std", f"{headline_emb.std():.4f}")
                                col3.metric("Min", f"{headline_emb.min():.4f}")
                                col4.metric("Max", f"{headline_emb.max():.4f}")
                        else:
                            headline_emb = embedder.encode([text_input], convert_to_numpy=True)
                        
                        # Step 3: Normalize headline embedding
                        if show_steps:
                            st.markdown("### 🔄 Step 3: L2 Normalization")
                            with st.expander("📊 Before & After Normalization", expanded=False):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("**Before Normalization:**")
                                    st.write(f"L2 Norm: {np.linalg.norm(headline_emb[0]):.4f}")
                                headline_emb_norm = normalize(headline_emb)
                                with col2:
                                    st.write("**After Normalization:**")
                                    st.write(f"L2 Norm: {np.linalg.norm(headline_emb_norm[0]):.4f}")
                                st.success("✅ Embedding normalized to unit length")
                        else:
                            headline_emb_norm = normalize(headline_emb)
                        
                        # Step 4: Generate placeholder features (dependency parsing & COMET)
                        if show_steps:
                            st.markdown("### 🧩 Step 4: Feature Concatenation")
                            st.info("""📌 **Note:** This demo uses simplified features. In production:
                            - Dependency parsing features would be extracted from CoNLL-U format
                            - COMET commonsense features would be generated using the COMET model
                            - For this demo, we're padding with zeros to match the expected dimension""")
                            
                            with st.expander("🔗 Feature Components", expanded=False):
                                st.write(f"1. **Headline Embeddings:** {headline_emb_norm.shape[1]} dims")
                                st.write(f"2. **Dependency Parse Features (placeholder):** {(1152 - headline_emb_norm.shape[1]) // 2} dims")
                                st.write(f"3. **COMET Features (placeholder):** {(1152 - headline_emb_norm.shape[1]) // 2} dims")
                                st.write(f"**Total Feature Vector:** 1152 dims")
                        
                        # Create full feature vector
                        if headline_emb_norm.shape[1] < 1152:
                            padding = np.zeros((1, 1152 - headline_emb_norm.shape[1]))
                            emb = np.hstack([headline_emb_norm, padding])
                            if show_steps:
                                with st.expander("➕ Padding Details", expanded=False):
                                    st.write(f"Original shape: {headline_emb_norm.shape}")
                                    st.write(f"Padding shape: {padding.shape}")
                                    st.write(f"Final shape: {emb.shape}")
                        else:
                            emb = headline_emb_norm
                        
                        # Step 5: Standardization
                        if show_steps:
                            st.markdown("### 📐 Step 5: Standardization (Scaling)")
                            with st.expander("⚖️ Before & After Scaling", expanded=False):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("**Before Scaling:**")
                                    st.write(f"Mean: {emb.mean():.4f}")
                                    st.write(f"Std: {emb.std():.4f}")
                                
                                if scaler is not None:
                                    emb_scaled = scaler.transform(emb)
                                    with col2:
                                        st.write("**After Scaling:**")
                                        st.write(f"Mean: {emb_scaled.mean():.4f}")
                                        st.write(f"Std: {emb_scaled.std():.4f}")
                                    st.success("✅ Features standardized (mean=0, std=1)")
                                else:
                                    emb_scaled = emb
                                    st.warning("⚠️ Scaler not loaded, using raw features")
                        else:
                            if scaler is not None:
                                emb_scaled = scaler.transform(emb)
                            else:
                                emb_scaled = emb
                        
                        # Step 6: Model Prediction
                        if show_steps:
                            st.markdown("### 🤖 Step 6: Model Inference")
                            st.write(f"**Model Type:** {model_type}")
                            st.write(f"**Input Shape:** {emb_scaled.shape}")
                        
                        if model_type == "ResidualMLP":
                            emb_tensor = torch.tensor(emb_scaled, dtype=torch.float32).to(device)
                            
                            if show_steps:
                                with st.expander("🧠 Model Forward Pass", expanded=False):
                                    st.write(f"**Device:** {device}")
                                    st.write(f"**Tensor Shape:** {emb_tensor.shape}")
                                    st.write(f"**Tensor dtype:** {emb_tensor.dtype}")
                            
                            with torch.no_grad():
                                logits = model(emb_tensor)
                                
                                if show_steps:
                                    with st.expander("📊 Model Output (Logits)", expanded=False):
                                        st.write(f"**Raw Logits:** {logits.cpu().numpy()[0]}")
                                        st.write(f"Class 0 (Non-Sarcastic): {logits[0][0].item():.4f}")
                                        st.write(f"Class 1 (Sarcastic): {logits[0][1].item():.4f}")
                                
                                probs = torch.softmax(logits, dim=1)
                                pred_class = probs.argmax(dim=1).item()
                                confidence = probs[0][pred_class].item()
                                
                                if show_steps:
                                    with st.expander("🎲 Softmax Probabilities", expanded=False):
                                        st.write(f"**After Softmax:**")
                                        st.write(f"P(Non-Sarcastic): {probs[0][0].item():.4f}")
                                        st.write(f"P(Sarcastic): {probs[0][1].item():.4f}")
                                        st.write(f"**Sum:** {probs[0].sum().item():.4f} (should be 1.0)")
                        else:
                            # MLPClassifier from sklearn
                            pred_class = model.predict(emb_scaled)[0]
                            probs = model.predict_proba(emb_scaled)[0]
                            confidence = probs[pred_class]
                        
                        # Display final result
                        st.markdown("---")
                        st.markdown("## 🎯 Final Prediction Result")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if pred_class == 1:
                                st.error("### 🎭 **SARCASTIC**")
                            else:
                                st.success("### 📰 **NON-SARCASTIC**")
                        
                        with col2:
                            st.metric("Confidence Score", f"{confidence*100:.2f}%", 
                                     delta=f"{abs(confidence - 0.5)*100:.1f}% from threshold" if confidence != 0.5 else None)
                        
                        # Probability distribution
                        st.markdown("### 📊 Probability Distribution")
                        prob_df = pd.DataFrame({
                            'Class': ['Non-Sarcastic (0)', 'Sarcastic (1)'],
                            'Probability': [probs[0] if model_type == "MLPClassifier" else probs[0][0].item(),
                                          probs[1] if model_type == "MLPClassifier" else probs[0][1].item()]
                        })
                        
                        fig, ax = plt.subplots(figsize=(10, 4))
                        colors = ['#2ca02c' if i == 0 else '#d62728' for i in range(2)]
                        bars = ax.barh(prob_df['Class'], prob_df['Probability'], color=colors)
                        ax.set_xlabel('Probability', fontsize=12)
                        ax.set_xlim(0, 1)
                        ax.set_title('Class Probabilities', fontsize=14, fontweight='bold')
                        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Decision Threshold')
                        
                        # Add value labels on bars
                        for i, (bar, prob) in enumerate(zip(bars, prob_df['Probability'])):
                            ax.text(prob + 0.02, i, f'{prob:.4f} ({prob*100:.2f}%)', 
                                   va='center', fontsize=10, fontweight='bold')
                        
                        ax.legend()
                        st.pyplot(fig)
                        
                        # Summary statistics
                        with st.expander("📈 Prediction Summary Statistics", expanded=False):
                            st.write(f"**Predicted Class:** {pred_class} ({'Sarcastic' if pred_class == 1 else 'Non-Sarcastic'})")
                            st.write(f"**Confidence:** {confidence:.6f}")
                            st.write(f"**Decision Certainty:** {abs(confidence - 0.5) * 2:.2%} (0% = uncertain, 100% = very certain)")
                            st.write(f"**Input Text Length:** {len(text_input)} characters")
                            st.write(f"**Feature Vector Size:** {emb_scaled.shape[1]} dimensions")
                        
                    except Exception as e:
                        st.error(f"❌ Error during prediction: {e}")
                        import traceback
                        with st.expander("🐛 Error Details", expanded=False):
                            st.code(traceback.format_exc(), language='python')
                        st.info("💡 Note: Full prediction requires COMET and dependency parsing features. This demo uses simplified features.")
            else:
                st.warning("⚠️ Please enter some text before predicting.")
    else:
        st.error("❌ Model or embedder not loaded. Please ensure model files are in the correct location.")
        st.info(f"🔍 Looking for model at: {MODEL_PATH}")
        st.info(f"🔍 Looking for scaler at: {SCALER_PATH}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    Tamil Sarcasm Detection System | Built with Streamlit, PyTorch & Transformers
</div>
""", unsafe_allow_html=True)
