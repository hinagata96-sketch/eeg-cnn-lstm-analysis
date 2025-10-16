import streamlit as st
import zipfile
import os
import shutil
import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy import signal
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler

# Set environment variables to handle TensorFlow mutex issues on macOS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
os.environ['GRPC_POLL_STRATEGY'] = 'poll'  # Handle gRPC issues
os.environ['TF_DISABLE_MKL'] = '1'  # Disable MKL to avoid threading issues
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Allow memory growth

# Try to import TensorFlow for CNN-LSTM
try:
    import tensorflow as tf
    # Configure TensorFlow for better compatibility
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.optimizers import Adam, RMSprop, SGD
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
    print(f"TensorFlow {tf.__version__} loaded successfully!")
except ImportError as e:
    TENSORFLOW_AVAILABLE = False
    print(f"TensorFlow import failed: {e}")
except Exception as e:
    TENSORFLOW_AVAILABLE = False
    print(f"TensorFlow configuration failed: {e}")

# Set page config
st.set_page_config(
    page_title="EEG Data Processing with ML",
    page_icon="🧠",
    layout="wide"
)

# Initialize session state
if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = None
if 'eeg_files' not in st.session_state:
    st.session_state.eeg_files = None
if 'channel_names' not in st.session_state:
    st.session_state.channel_names = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

# Functions from your original code
def extract_zip(zip_path, extract_dir):
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    # Find main data folder
    for item in os.listdir(extract_dir):
        if item.startswith('.') or item == '__MACOSX':
            continue
        candidate = os.path.join(extract_dir, item)
        if os.path.isdir(candidate):
            return candidate
    raise Exception("No valid data folder found!")

def load_eeg_csvs(data_root):
    """Return: dict {class: [csv_paths]}"""
    eeg_files = {}
    for class_folder in sorted(os.listdir(data_root)):
        if class_folder.startswith('.') or class_folder == '__MACOSX':
            continue
        class_path = os.path.join(data_root, class_folder)
        if os.path.isdir(class_path):
            eeg_files[class_folder] = []
            for csv_file in sorted(os.listdir(class_path)):
                if csv_file.endswith('.csv'):
                    eeg_files[class_folder].append(os.path.join(class_path, csv_file))
    return eeg_files

def detect_artifact_components(ica_components, threshold_percentile=95):
    artifact_indices = []
    n_components = ica_components.shape[0]
    for i in range(n_components):
        component = ica_components[i, :]
        kurtosis = np.mean((component - np.mean(component))**4) / (np.std(component)**4)
        skewness = np.mean((component - np.mean(component))**3) / (np.std(component)**3)
        variance = np.var(component)
        freqs, psd = signal.welch(component, fs=250, nperseg=min(1024, len(component)//4))
        high_freq_power = np.sum(psd[freqs > 30]) / np.sum(psd)
        is_artifact = (
            kurtosis > 5 or
            abs(skewness) > 2 or
            variance > np.percentile([np.var(ica_components[j, :]) for j in range(n_components)], threshold_percentile) or
            high_freq_power > 0.6
        )
        if is_artifact:
            artifact_indices.append(i)
    return artifact_indices

def apply_ica_preprocessing(data, n_components=None, remove_artifacts=True, fs=250):
    # data: [timesteps, channels]
    n_timepoints, n_channels = data.shape
    if n_components is None:
        n_components = min(n_channels, n_timepoints//100)
    b, a = signal.butter(4, 1.0/(fs/2), btype='high')
    data_filtered = signal.filtfilt(b, a, data, axis=0)
    ica = FastICA(n_components=n_components, random_state=42, max_iter=1000, tol=1e-4)
    try:
        ica_components = ica.fit_transform(data_filtered)
        mixing_matrix = ica.mixing_
        artifact_indices = detect_artifact_components(ica_components.T) if remove_artifacts else []
        ica_components_clean = ica_components.copy()
        if artifact_indices:
            ica_components_clean[:, artifact_indices] = 0
        data_clean = ica_components_clean @ mixing_matrix.T
        if data_clean.shape != data.shape:
            data_clean = data_clean[:, :n_channels]
        return data_clean, {'artifact_indices': artifact_indices}
    except Exception as e:
        st.error(f"ICA failed: {e}")
        return data, {'artifact_indices': []}

def identify_channels(csv_path):
    df = pd.read_csv(csv_path)
    channel_names = list(df.columns)
    return channel_names

# CNN-LSTM specific functions
def prepare_cnn_lstm_data(eeg_files, selected_indices, desired_timesteps=5000):
    """Prepare data for CNN-LSTM model"""
    data = []
    labels = []
    class_names = list(eeg_files.keys())
    
    for class_idx, class_name in enumerate(class_names):
        for csv_path in eeg_files[class_name]:
            df = pd.read_csv(csv_path)
            arr = df.values.astype(np.float32)[:, selected_indices]
            
            # Apply ICA preprocessing
            arr_clean, _ = apply_ica_preprocessing(arr, remove_artifacts=True)
            
            # Pad or truncate to fixed timesteps
            if arr_clean.shape[0] < desired_timesteps:
                pad_width = desired_timesteps - arr_clean.shape[0]
                arr_clean = np.pad(arr_clean, ((0, pad_width), (0, 0)), mode='constant')
            elif arr_clean.shape[0] > desired_timesteps:
                arr_clean = arr_clean[:desired_timesteps, :]
            
            data.append(arr_clean)
            labels.append(class_idx)
    
    X = np.stack(data)  # shape: (samples, timesteps, features)
    y = np.array(labels)
    return X, y, class_names

def build_cnn_lstm_model(input_shape, num_classes, cnn_config, lstm_config, dense_config, optimizer_config):
    """Build CNN-LSTM model with user-defined hyperparameters"""
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is required for CNN-LSTM model")
    
    model = Sequential()
    
    # Add CNN layers
    for i, layer_config in enumerate(cnn_config):
        if i == 0:
            # First CNN layer
            model.add(Conv1D(
                filters=layer_config['filters'],
                kernel_size=layer_config['kernel_size'],
                activation=layer_config['activation'],
                padding=layer_config['padding'],
                input_shape=input_shape
            ))
        else:
            model.add(Conv1D(
                filters=layer_config['filters'],
                kernel_size=layer_config['kernel_size'],
                activation=layer_config['activation'],
                padding=layer_config['padding']
            ))
        
        # Add batch normalization if specified
        if layer_config.get('batch_norm', False):
            model.add(BatchNormalization())
        
        # Add pooling if specified
        if layer_config.get('pool_size', 0) > 1:
            model.add(MaxPooling1D(pool_size=layer_config['pool_size']))
        
        # Add dropout if specified
        if layer_config.get('dropout', 0) > 0:
            model.add(Dropout(layer_config['dropout']))
    
    # Add LSTM layers
    for i, layer_config in enumerate(lstm_config):
        return_sequences = (i < len(lstm_config) - 1)  # Return sequences for all but last LSTM
        
        model.add(LSTM(
            units=layer_config['units'],
            return_sequences=return_sequences,
            dropout=layer_config.get('dropout', 0),
            recurrent_dropout=layer_config.get('recurrent_dropout', 0)
        ))
        
        if layer_config.get('batch_norm', False):
            model.add(BatchNormalization())
    
    # Add Dense layers
    for layer_config in dense_config:
        model.add(Dense(
            units=layer_config['units'],
            activation=layer_config['activation']
        ))
        
        if layer_config.get('dropout', 0) > 0:
            model.add(Dropout(layer_config['dropout']))
    
    # Output layer
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile model
    optimizer_name = optimizer_config['name']
    learning_rate = optimizer_config['learning_rate']
    
    if optimizer_name == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = SGD(learning_rate=learning_rate, momentum=optimizer_config.get('momentum', 0.0))
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Feature extraction functions
def extract_statistical_features(data):
    """Extract statistical features from EEG data"""
    features = []
    # Basic statistical features
    features.extend([
        np.mean(data, axis=0),
        np.std(data, axis=0),
        np.var(data, axis=0),
        np.min(data, axis=0),
        np.max(data, axis=0),
        np.median(data, axis=0)
    ])
    return np.concatenate(features)

def extract_frequency_features(data, fs=250):
    """Extract frequency domain features"""
    features = []
    for channel in range(data.shape[1]):
        freqs, psd = signal.welch(data[:, channel], fs=fs, nperseg=min(1024, len(data)//4))
        
        # Frequency band powers
        delta_power = np.sum(psd[(freqs >= 0.5) & (freqs < 4)])
        theta_power = np.sum(psd[(freqs >= 4) & (freqs < 8)])
        alpha_power = np.sum(psd[(freqs >= 8) & (freqs < 13)])
        beta_power = np.sum(psd[(freqs >= 13) & (freqs < 30)])
        gamma_power = np.sum(psd[(freqs >= 30) & (freqs < 100)])
        
        features.extend([delta_power, theta_power, alpha_power, beta_power, gamma_power])
    
    return np.array(features)

def prepare_ml_data(eeg_files, selected_indices, max_samples_per_class=None):
    """Prepare data for traditional ML models"""
    X = []
    y = []
    class_names = list(eeg_files.keys())
    
    for class_idx, class_name in enumerate(class_names):
        class_samples = 0
        for csv_path in eeg_files[class_name]:
            if max_samples_per_class and class_samples >= max_samples_per_class:
                break
                
            df = pd.read_csv(csv_path)
            data = df.values.astype(np.float32)[:, selected_indices]
            
            # Apply ICA preprocessing
            data_clean, _ = apply_ica_preprocessing(data, remove_artifacts=True)
            
            # Extract features
            stat_features = extract_statistical_features(data_clean)
            freq_features = extract_frequency_features(data_clean)
            
            # Combine features
            combined_features = np.concatenate([stat_features, freq_features])
            
            X.append(combined_features)
            y.append(class_idx)
            class_samples += 1
    
    return np.array(X), np.array(y), class_names

# Streamlit App
def main():
    st.title("🧠 EEG Data Processing with Machine Learning")
    st.markdown("### Compatible with macOS Sequoia and Python 3.13")
    st.markdown("---")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    step = st.sidebar.radio("Select Step:", [
        "1. Upload Data",
        "2. Channel Selection", 
        "3. ICA Processing",
        "4. CNN-LSTM Model Training",
        "5. Visualization"
    ])
    
    if step == "1. Upload Data":
        st.header("Step 1: Upload EEG Data")
        
        uploaded_file = st.file_uploader("Choose a ZIP file containing EEG data", type="zip")
        
        if uploaded_file is not None:
            with st.spinner("Extracting ZIP file..."):
                # Save uploaded file to temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    zip_path = tmp_file.name
                
                try:
                    # Extract zip
                    extract_dir = tempfile.mkdtemp()
                    data_root = extract_zip(zip_path, extract_dir)
                    eeg_files = load_eeg_csvs(data_root)
                    
                    # Store in session state
                    st.session_state.extracted_data = data_root
                    st.session_state.eeg_files = eeg_files
                    
                    # Display results
                    st.success("✅ ZIP file extracted successfully!")
                    
                    classes = list(eeg_files.keys())
                    st.write(f"**Classes found:** {classes}")
                    
                    for class_name in classes:
                        st.write(f"- **{class_name}**: {len(eeg_files[class_name])} files")
                    
                    # Load first file to get channel info
                    if classes:
                        first_file = eeg_files[classes[0]][0]
                        channel_names = identify_channels(first_file)
                        st.session_state.channel_names = channel_names
                        st.write(f"**Channels detected:** {len(channel_names)} channels")
                        
                except Exception as e:
                    st.error(f"Error processing ZIP file: {e}")
                finally:
                    # Clean up
                    os.unlink(zip_path)
    
    elif step == "2. Channel Selection":
        st.header("Step 2: Select EEG Channels")
        
        if st.session_state.channel_names is None:
            st.warning("Please upload data first!")
            return
        
        channel_names = st.session_state.channel_names
        st.write(f"**Available channels ({len(channel_names)}):**")
        
        # Create columns for better layout
        cols = st.columns(3)
        for i, name in enumerate(channel_names):
            cols[i % 3].write(f"{i}: {name}")
        
        st.markdown("---")
        
        # Channel selection
        selected_indices = st.multiselect(
            "Select channels:",
            options=list(range(len(channel_names))),
            format_func=lambda x: f"{x}: {channel_names[x]}",
            default=list(range(min(8, len(channel_names))))  # Default to first 8 channels
        )
        
        if selected_indices:
            selected_channels = [channel_names[i] for i in selected_indices]
            st.write(f"**Selected channels:** {selected_channels}")
            st.session_state.selected_indices = selected_indices
            st.session_state.selected_channels = selected_channels
            st.success(f"✅ Selected {len(selected_indices)} channels")
    
    elif step == "3. ICA Processing":
        st.header("Step 3: ICA Artifact Removal")
        
        if st.session_state.eeg_files is None or not hasattr(st.session_state, 'selected_indices'):
            st.warning("Please complete previous steps first!")
            return
        
        eeg_files = st.session_state.eeg_files
        selected_indices = st.session_state.selected_indices
        
        # ICA parameters
        st.subheader("ICA Parameters")
        col1, col2 = st.columns(2)
        remove_artifacts = col1.checkbox("Remove artifacts", value=True)
        fs = col2.number_input("Sampling frequency (Hz):", value=250, min_value=1)
        
        if st.button("Start ICA Processing"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_files = sum(len(files) for files in eeg_files.values())
            processed = 0
            
            processed_data = {}
            
            for class_name in eeg_files.keys():
                status_text.text(f"Processing class: {class_name}")
                processed_data[class_name] = []
                
                for csv_path in eeg_files[class_name]:
                    status_text.text(f"Processing: {os.path.basename(csv_path)}")
                    
                    try:
                        # Load and process data
                        df = pd.read_csv(csv_path)
                        arr = df.values.astype(np.float32)
                        arr_selected = arr[:, selected_indices]
                        
                        # Apply ICA
                        arr_clean, info = apply_ica_preprocessing(
                            arr_selected, 
                            remove_artifacts=remove_artifacts, 
                            fs=fs
                        )
                        
                        # Store processed data
                        processed_data[class_name].append({
                            'file_path': csv_path,
                            'clean_data': arr_clean,
                            'artifact_info': info
                        })
                        
                    except Exception as e:
                        st.error(f"Error processing {csv_path}: {e}")
                    
                    processed += 1
                    progress_bar.progress(processed / total_files)
            
            # Store processed data in session state
            st.session_state.processed_data = processed_data
            
            status_text.text("✅ ICA processing completed!")
            st.success("All files processed successfully!")
    
    elif step == "4. CNN-LSTM Model Training":
        st.header("Step 4: CNN-LSTM Model Training")
        
        if not TENSORFLOW_AVAILABLE:
            st.error("❌ TensorFlow is not available. Please install TensorFlow to use CNN-LSTM functionality.")
            st.code("pip install tensorflow")
            return
        
        if st.session_state.eeg_files is None or not hasattr(st.session_state, 'selected_indices'):
            st.warning("Please complete previous steps first!")
            return
        
        eeg_files = st.session_state.eeg_files
        selected_indices = st.session_state.selected_indices
        
        # Data Configuration
        st.subheader("📊 Data Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            desired_timesteps = st.number_input("Sequence length (timesteps):", value=5000, min_value=1000, max_value=50000, step=1000)
            epochs = st.number_input("Training epochs:", value=50, min_value=5, max_value=200)
        
        with col2:
            batch_size = st.selectbox("Batch size:", [4, 8, 16, 32], index=1)
            test_size = st.slider("Test set size:", 0.1, 0.4, 0.2, 0.05)
        
        # CNN Architecture Configuration
        st.subheader("🔥 CNN Layer Configuration")
        num_cnn_layers = st.selectbox("Number of CNN layers:", [1, 2, 3, 4], index=1)
        
        cnn_config = []
        for i in range(num_cnn_layers):
            st.write(f"**CNN Layer {i+1}:**")
            col1, col2, col3, col4 = st.columns(4)
            
            filters = col1.number_input(
                f"Filters:", 
                value=32*(2**i), 
                min_value=8, 
                max_value=512, 
                step=8, 
                key=f"cnn_filters_{i}"
            )
            
            kernel_size = col2.number_input(
                f"Kernel size:", 
                value=5, 
                min_value=3, 
                max_value=15, 
                step=2, 
                key=f"cnn_kernel_{i}"
            )
            
            activation = col3.selectbox(
                f"Activation:", 
                ['relu', 'tanh', 'sigmoid', 'elu'], 
                index=0, 
                key=f"cnn_activation_{i}"
            )
            
            padding = col4.selectbox(
                f"Padding:", 
                ['same', 'valid'], 
                index=0, 
                key=f"cnn_padding_{i}"
            )
            
            # Additional CNN options
            col5, col6, col7, col8 = st.columns(4)
            
            pool_size = col5.number_input(
                f"Pool size:", 
                value=2 if i < 2 else 1, 
                min_value=1, 
                max_value=5, 
                key=f"cnn_pool_{i}"
            )
            
            dropout = col6.slider(
                f"Dropout:", 
                0.0, 0.8, 0.2, 0.1, 
                key=f"cnn_dropout_{i}"
            )
            
            batch_norm = col7.checkbox(
                f"Batch Norm", 
                value=True, 
                key=f"cnn_bn_{i}"
            )
            
            cnn_config.append({
                'filters': filters,
                'kernel_size': kernel_size,
                'activation': activation,
                'padding': padding,
                'pool_size': pool_size,
                'dropout': dropout,
                'batch_norm': batch_norm
            })
        
        # LSTM Architecture Configuration  
        st.subheader("🧠 LSTM Layer Configuration")
        num_lstm_layers = st.selectbox("Number of LSTM layers:", [1, 2, 3], index=0)
        
        lstm_config = []
        for i in range(num_lstm_layers):
            st.write(f"**LSTM Layer {i+1}:**")
            col1, col2, col3, col4 = st.columns(4)
            
            units = col1.number_input(
                f"Units:", 
                value=64, 
                min_value=16, 
                max_value=512, 
                step=16, 
                key=f"lstm_units_{i}"
            )
            
            dropout = col2.slider(
                f"Dropout:", 
                0.0, 0.8, 0.2, 0.1, 
                key=f"lstm_dropout_{i}"
            )
            
            recurrent_dropout = col3.slider(
                f"Recurrent Dropout:", 
                0.0, 0.8, 0.1, 0.1, 
                key=f"lstm_rec_dropout_{i}"
            )
            
            batch_norm = col4.checkbox(
                f"Batch Norm", 
                value=False, 
                key=f"lstm_bn_{i}"
            )
            
            lstm_config.append({
                'units': units,
                'dropout': dropout,
                'recurrent_dropout': recurrent_dropout,
                'batch_norm': batch_norm
            })
        
        # Dense Layer Configuration
        st.subheader("⚡ Dense Layer Configuration")
        num_dense_layers = st.selectbox("Number of Dense layers:", [0, 1, 2], index=1)
        
        dense_config = []
        for i in range(num_dense_layers):
            col1, col2, col3 = st.columns(3)
            
            units = col1.number_input(
                f"Dense {i+1} Units:", 
                value=64, 
                min_value=8, 
                max_value=512, 
                step=8, 
                key=f"dense_units_{i}"
            )
            
            activation = col2.selectbox(
                f"Dense {i+1} Activation:", 
                ['relu', 'tanh', 'sigmoid', 'elu'], 
                index=0, 
                key=f"dense_activation_{i}"
            )
            
            dropout = col3.slider(
                f"Dense {i+1} Dropout:", 
                0.0, 0.8, 0.3, 0.1, 
                key=f"dense_dropout_{i}"
            )
            
            dense_config.append({
                'units': units,
                'activation': activation,
                'dropout': dropout
            })
        
        # Optimizer Configuration
        st.subheader("🎯 Optimizer Configuration")
        col1, col2, col3 = st.columns(3)
        
        optimizer_name = col1.selectbox("Optimizer:", ['adam', 'rmsprop', 'sgd'], index=0)
        learning_rate = col2.number_input("Learning rate:", value=0.001, min_value=0.0001, max_value=0.1, format="%.4f")
        
        optimizer_config = {
            'name': optimizer_name,
            'learning_rate': learning_rate
        }
        
        if optimizer_name == 'sgd':
            momentum = col3.slider("Momentum:", 0.0, 0.99, 0.9, 0.01)
            optimizer_config['momentum'] = momentum
        
        # Training Configuration
        st.subheader("🏃 Training Configuration")
        col1, col2 = st.columns(2)
        
        early_stopping = col1.checkbox("Early Stopping", value=True)
        reduce_lr = col2.checkbox("Reduce LR on Plateau", value=True)
        
        if early_stopping:
            patience = col1.number_input("Early Stopping Patience:", value=10, min_value=3, max_value=50)
        
        if reduce_lr:
            lr_patience = col2.number_input("LR Reduction Patience:", value=5, min_value=2, max_value=20)
        
        # Start Training Button
        if st.button("🚀 Start CNN-LSTM Training"):
            with st.spinner("Preparing data and training CNN-LSTM model..."):
                try:
                    # Prepare data
                    X, y, class_names = prepare_cnn_lstm_data(eeg_files, selected_indices, desired_timesteps)
                    num_classes = len(class_names)
                    y_cat = to_categorical(y, num_classes)
                    
                    st.write(f"**Data prepared:**")
                    st.write(f"- Input shape: {X.shape}")
                    st.write(f"- Classes: {class_names}")
                    st.write(f"- Samples per class: {np.bincount(y)}")
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y_cat, test_size=test_size, random_state=42, stratify=y
                    )
                    
                    # Build model
                    model = build_cnn_lstm_model(
                        input_shape=(X.shape[1], X.shape[2]),
                        num_classes=num_classes,
                        cnn_config=cnn_config,
                        lstm_config=lstm_config,
                        dense_config=dense_config,
                        optimizer_config=optimizer_config
                    )
                    
                    # Display model summary
                    st.subheader("🏗️ Model Architecture")
                    model_summary = []
                    model.summary(print_fn=lambda x: model_summary.append(x))
                    st.text('\n'.join(model_summary))
                    
                    # Prepare callbacks
                    callbacks = []
                    if early_stopping:
                        callbacks.append(EarlyStopping(
                            monitor='val_loss',
                            patience=patience,
                            restore_best_weights=True
                        ))
                    
                    if reduce_lr:
                        callbacks.append(ReduceLROnPlateau(
                            monitor='val_loss',
                            factor=0.5,
                            patience=lr_patience,
                            min_lr=0.00001
                        ))
                    
                    # Train model
                    st.subheader("📈 Training Progress")
                    progress_container = st.container()
                    
                    # Create progress tracking
                    progress_bar = st.progress(0)
                    metrics_container = st.container()
                    
                    class StreamlitCallback(tf.keras.callbacks.Callback):
                        def __init__(self, progress_bar, metrics_container, total_epochs):
                            self.progress_bar = progress_bar
                            self.metrics_container = metrics_container
                            self.total_epochs = total_epochs
                            self.epoch_metrics = []
                        
                        def on_epoch_end(self, epoch, logs=None):
                            progress = (epoch + 1) / self.total_epochs
                            self.progress_bar.progress(progress)
                            
                            if logs:
                                self.epoch_metrics.append(logs)
                                with self.metrics_container:
                                    st.write(f"Epoch {epoch + 1}/{self.total_epochs}")
                                    col1, col2, col3, col4 = st.columns(4)
                                    col1.metric("Loss", f"{logs.get('loss', 0):.4f}")
                                    col2.metric("Accuracy", f"{logs.get('accuracy', 0):.4f}")
                                    col3.metric("Val Loss", f"{logs.get('val_loss', 0):.4f}")
                                    col4.metric("Val Accuracy", f"{logs.get('val_accuracy', 0):.4f}")
                    
                    streamlit_callback = StreamlitCallback(progress_bar, metrics_container, epochs)
                    callbacks.append(streamlit_callback)
                    
                    # Train the model
                    history = model.fit(
                        X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=callbacks,
                        verbose=0
                    )
                    
                    # Store results in session state
                    st.session_state.cnn_lstm_model = model
                    st.session_state.cnn_lstm_history = history
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test
                    st.session_state.class_names = class_names
                    
                    # Display final results
                    st.success("✅ CNN-LSTM model training completed!")
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    y_pred_classes = np.argmax(y_pred, axis=1)
                    y_test_classes = np.argmax(y_test, axis=1)
                    
                    final_accuracy = accuracy_score(y_test_classes, y_pred_classes)
                    
                    st.subheader("🎯 Final Results")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Test Accuracy", f"{final_accuracy:.4f}")
                    col2.metric("Final Train Loss", f"{history.history['loss'][-1]:.4f}")
                    col3.metric("Final Val Loss", f"{history.history['val_loss'][-1]:.4f}")
                    
                    # Plot training history
                    st.subheader("📊 Training History")
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                    
                    # Accuracy plot
                    ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
                    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
                    ax1.set_title('Model Accuracy')
                    ax1.set_ylabel('Accuracy')
                    ax1.set_xlabel('Epoch')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    # Loss plot
                    ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
                    ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
                    ax2.set_title('Model Loss')
                    ax2.set_ylabel('Loss')
                    ax2.set_xlabel('Epoch')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                    
                    # Classification report
                    st.subheader("📋 Classification Report")
                    report = classification_report(y_test_classes, y_pred_classes, target_names=class_names)
                    st.text(report)
                    
                    # Confusion matrix
                    st.subheader("🔥 Confusion Matrix")
                    cm = confusion_matrix(y_test_classes, y_pred_classes)
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                               xticklabels=class_names, yticklabels=class_names, ax=ax)
                    ax.set_ylabel('True Label')
                    ax.set_xlabel('Predicted Label')
                    ax.set_title('Confusion Matrix')
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"❌ Error during CNN-LSTM training: {str(e)}")
                    st.write("**Debug info:**")
                    st.write(f"TensorFlow available: {TENSORFLOW_AVAILABLE}")
                    if TENSORFLOW_AVAILABLE:
                        st.write(f"TensorFlow version: {tf.__version__}")
    
    elif step == "5. Visualization":
        st.header("Step 5: Data Visualization")
        
        if st.session_state.eeg_files is None:
            st.warning("Please upload data first!")
            return
        
        eeg_files = st.session_state.eeg_files
        classes = list(eeg_files.keys())
        
        # File selection for visualization
        selected_class = st.selectbox("Select class:", classes)
        available_files = [os.path.basename(f) for f in eeg_files[selected_class]]
        selected_file_name = st.selectbox("Select file:", available_files)
        
        if st.button("Visualize Data"):
            # Get full path
            selected_file_path = None
            for file_path in eeg_files[selected_class]:
                if os.path.basename(file_path) == selected_file_name:
                    selected_file_path = file_path
                    break
            
            if selected_file_path:
                # Load data
                df = pd.read_csv(selected_file_path)
                
                # Plot original data
                st.subheader("Original EEG Data")
                fig, ax = plt.subplots(figsize=(12, 8))
                
                if hasattr(st.session_state, 'selected_indices'):
                    selected_indices = st.session_state.selected_indices
                    data_to_plot = df.iloc[:, selected_indices]
                    channel_names_to_plot = st.session_state.selected_channels
                else:
                    data_to_plot = df.iloc[:, :min(8, len(df.columns))]  # Plot first 8 channels
                    channel_names_to_plot = list(df.columns)[:min(8, len(df.columns))]
                
                for i, channel in enumerate(channel_names_to_plot):
                    ax.plot(data_to_plot.iloc[:, i] + i*100, label=channel)
                
                ax.set_xlabel("Time (samples)")
                ax.set_ylabel("Amplitude (offset)")
                ax.set_title(f"EEG Data: {selected_file_name}")
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                
                st.pyplot(fig)
                
                # Show data statistics
                st.subheader("Data Statistics")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Shape:**", df.shape)
                    st.write("**Duration:**", f"{df.shape[0]/250:.2f} seconds (assuming 250 Hz)")
                
                with col2:
                    st.write("**Channels:**", len(df.columns))
                    st.write("**Missing values:**", df.isnull().sum().sum())
                
                # Model performance visualization (if model exists)
                if hasattr(st.session_state, 'model') and hasattr(st.session_state, 'y_test'):
                    st.subheader("Model Performance")
                    
                    # Accuracy by class
                    y_test = st.session_state.y_test
                    y_pred = st.session_state.y_pred
                    class_names = st.session_state.class_names
                    
                    class_accuracies = []
                    for i, class_name in enumerate(class_names):
                        mask = y_test == i
                        if np.sum(mask) > 0:
                            class_acc = accuracy_score(y_test[mask], y_pred[mask])
                            class_accuracies.append(class_acc)
                        else:
                            class_accuracies.append(0)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.bar(class_names, class_accuracies)
                    ax.set_ylabel('Accuracy')
                    ax.set_title('Accuracy by Class')
                    ax.set_ylim(0, 1)
                    
                    # Add value labels on bars
                    for bar, acc in zip(bars, class_accuracies):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{acc:.3f}', ha='center', va='bottom')
                    
                    st.pyplot(fig)

if __name__ == "__main__":
    main()