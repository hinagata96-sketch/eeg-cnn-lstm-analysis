# 🧠 EEG CNN-LSTM Analysis Platform

A comprehensive web application for EEG signal processing and classification using Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks built with PyTorch and Streamlit.

## ✨ Features

### 🔄 Complete EEG Processing Pipeline
- **Data Upload**: ZIP file support for organized EEG datasets
- **Channel Selection**: Interactive EEG channel selection interface
- **ICA Preprocessing**: Independent Component Analysis for artifact removal
- **CNN-LSTM Training**: Advanced deep learning model with customizable architecture
- **Model Testing**: Comprehensive evaluation on unseen data

### 🎛️ Advanced Configuration
- **Flexible Data Splits**: Customizable train/validation/test ratios
- **LSTM Hyperparameters**: Multi-layer, bidirectional, dropout options
- **CNN Architecture**: Configurable layers, filters, and kernel sizes
- **Real-time Training**: Live graphs showing accuracy and loss progression

### 💾 Model Management
- **Save/Load Models**: Persistent model storage with full configuration
- **Test on New Data**: Upload and classify completely new EEG recordings
- **Performance Metrics**: Confusion matrices, classification reports, accuracy metrics

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/eeg-cnn-lstm.git
   cd eeg-cnn-lstm
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run streamlit_app.py
   ```

5. **Open your browser** to `http://localhost:8501`

## 📊 Usage

### Step 1: Upload Data
- Prepare your EEG data in ZIP format with class folders
- Each CSV file represents one EEG recording sample
- Structure: `ZIP → Class_Folders → CSV_Files`

### Step 2: Channel Selection
- Select relevant EEG channels for analysis
- Default: First 8 channels automatically selected

### Step 3: ICA Processing
- Apply Independent Component Analysis
- Remove artifacts automatically
- Configure sampling frequency

### Step 4: CNN-LSTM Training
- **Data Split Configuration**: Set custom train/val/test ratios
- **CNN Layers**: Configure filters, kernel sizes, pooling
- **LSTM Parameters**: Units, layers, dropout, bidirectional options
- **Training**: Real-time visualization of training progress

### Step 5: Model Testing
- **Save Models**: Store trained models with full configuration
- **Load Models**: Restore previously saved models
- **Test on Holdout**: Evaluate on reserved test set
- **New Data Testing**: Upload and classify new EEG recordings

## 🏗️ Architecture

### CNN-LSTM Model Structure
```
Input EEG Data (samples, timesteps, channels)
    ↓
CNN Feature Extraction
    ├── Conv1D → ReLU → MaxPool → Dropout
    └── Conv1D → ReLU → MaxPool → Dropout
    ↓
LSTM Temporal Processing
    └── Multi-layer LSTM (optional bidirectional)
    ↓
Classification Head
    ├── Dropout → Dense → ReLU
    └── Dense → Softmax → Class Predictions
```

### Key Components
- **PyTorch**: Deep learning framework
- **Streamlit**: Web application interface
- **scikit-learn**: Preprocessing and evaluation
- **matplotlib/seaborn**: Visualization

## 📈 Data Format

### Expected Input
- **Format**: ZIP file containing class folders
- **Structure**: 
  ```
  dataset.zip
  ├── class1/
  │   ├── sample1.csv
  │   ├── sample2.csv
  │   └── ...
  └── class2/
      ├── sample1.csv
      ├── sample2.csv
      └── ...
  ```
- **CSV Format**: Each row = timepoint, each column = EEG channel
- **Recommended**: 90,000 timepoints per file (6 minutes at 250 Hz)

## 🛠️ Technical Details

### Requirements
- **streamlit**: Web application framework
- **torch**: PyTorch deep learning
- **scikit-learn**: Machine learning utilities
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **matplotlib**: Plotting
- **seaborn**: Statistical visualization
- **scipy**: Scientific computing

### Model Capabilities
- **Multi-class Classification**: Support for any number of EEG classes
- **Temporal Learning**: LSTM networks for sequential pattern recognition
- **Spatial Features**: CNN for channel-wise feature extraction
- **Customizable Architecture**: Flexible layer configuration
- **Real-time Training**: Live progress monitoring

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

- **Project Link**: [https://github.com/your-username/eeg-cnn-lstm](https://github.com/your-username/eeg-cnn-lstm)
- **Issues**: [https://github.com/your-username/eeg-cnn-lstm/issues](https://github.com/your-username/eeg-cnn-lstm/issues)

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- Streamlit for the intuitive web application framework
- EEG research community for methodological insights

---
*Built with ❤️ for EEG signal analysis and neuroscience research*