import streamlit as st
import zipfile
import os
import shutil
import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from scipy import signal
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
# PyTorch imports for CNN-LSTM
import torch
import torch.nn as nn
import torch.optim as optim

# Set page config
st.set_page_config(
    page_title="EEG Data Processing with CNN-LSTM",
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

# CNN-LSTM Functions
def prepare_cnn_lstm_data(eeg_files, selected_indices, desired_timesteps=90000):
    """Return: X (samples, timesteps, features), y (labels), class_names"""
    data = []
    labels = []
    class_names = list(eeg_files.keys())
    for class_idx, class_name in enumerate(class_names):
        for csv_path in eeg_files[class_name]:
            df = pd.read_csv(csv_path)
            arr = df.values.astype(np.float32)[:, selected_indices]
            # Pad or truncate to fixed timesteps
            if arr.shape[0] < desired_timesteps:
                pad_width = desired_timesteps - arr.shape[0]
                arr = np.pad(arr, ((0, pad_width), (0, 0)), mode='constant')
            elif arr.shape[0] > desired_timesteps:
                arr = arr[:desired_timesteps, :]
            data.append(arr)
            labels.append(class_idx)
    X = np.stack(data)  # shape: (samples, timesteps, features)
    y = np.array(labels)
    return X, y, class_names

def build_cnn_lstm_model(input_shape, num_classes, cnn_config=None, lstm_units=64, dense_units=64, dropout_rate=0.3, num_lstm_layers=1, lstm_dropout=0.0, bidirectional=False, lstm_bias=True):
    """Build a simple CNN-LSTM model."""
    if cnn_config is None:
        cnn_config = [
            {'filters': 32, 'kernel_size': 5, 'pool_size': 2},
            {'filters': 64, 'kernel_size': 5, 'pool_size': 2}
        ]
    
    class CNNLSTM(nn.Module):
        def __init__(self, input_shape, num_classes, cnn_config, lstm_units, dense_units, dropout_rate, num_lstm_layers, lstm_dropout, bidirectional, lstm_bias):
            super(CNNLSTM, self).__init__()
            self.bidirectional = bidirectional
            self.cnn_layers = nn.Sequential(
                nn.Conv1d(input_shape[1], cnn_config[0]['filters'], kernel_size=cnn_config[0]['kernel_size'], padding=cnn_config[0].get('padding', 0)),
                nn.ReLU(),
                nn.MaxPool1d(cnn_config[0]['pool_size']),
                nn.Dropout(dropout_rate),
                nn.Conv1d(cnn_config[0]['filters'], cnn_config[1]['filters'], kernel_size=cnn_config[1]['kernel_size'], padding=cnn_config[1].get('padding', 0)),
                nn.ReLU(),
                nn.MaxPool1d(cnn_config[1]['pool_size']),
                nn.Dropout(dropout_rate)
            )
            self.lstm = nn.LSTM(
                input_size=cnn_config[1]['filters'], 
                hidden_size=lstm_units, 
                num_layers=num_lstm_layers,
                dropout=lstm_dropout if num_lstm_layers > 1 else 0.0,
                bidirectional=bidirectional,
                bias=lstm_bias,
                batch_first=True
            )
            
            # Adjust linear layer input size for bidirectional LSTM
            lstm_output_size = lstm_units * 2 if bidirectional else lstm_units
            self.fc1 = nn.Linear(lstm_output_size, dense_units)
            self.fc2 = nn.Linear(dense_units, num_classes)
            self.dropout = nn.Dropout(dropout_rate)
            
        def forward(self, x):
            # x shape: (batch, timesteps, features)
            x = x.permute(0, 2, 1)  # PyTorch expects (batch, channels, timesteps)
            x = self.cnn_layers(x)
            x = x.permute(0, 2, 1)  # (batch, timesteps, channels)
            out, (h_n, c_n) = self.lstm(x)
            
            # For bidirectional LSTM, concatenate final forward and backward hidden states
            if self.bidirectional:
                # h_n shape: (num_layers * 2, batch, hidden_size)
                forward_hidden = h_n[-2]  # Last layer forward
                backward_hidden = h_n[-1]  # Last layer backward
                x = torch.cat((forward_hidden, backward_hidden), dim=1)
            else:
                x = h_n[-1]  # Last layer hidden state
            
            x = self.dropout(x)
            x = self.fc1(x)
            x = torch.relu(x)
            x = self.fc2(x)
            return x
    
    model = CNNLSTM(input_shape, num_classes, cnn_config, lstm_units, dense_units, dropout_rate, num_lstm_layers, lstm_dropout, bidirectional, lstm_bias)
    return model

# Streamlit App
def main():
    st.title("🧠 EEG Data Processing with CNN-LSTM")
    st.markdown("---")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    step = st.sidebar.radio("Select Step:", [
        "1. Upload Data",
        "2. Channel Selection", 
        "3. ICA Processing",
        "4. CNN-LSTM Training",
        "5. Model Testing"
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
            total_artifacts_removed = 0
            
            for class_name in eeg_files.keys():
                status_text.text(f"Processing class: {class_name}")
                
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
                        
                        # Count artifacts (but don't display individual files)
                        artifact_count = len(info['artifact_indices'])
                        total_artifacts_removed += artifact_count
                        
                    except Exception as e:
                        st.error(f"Error processing {csv_path}: {e}")
                    
                    processed += 1
                    progress_bar.progress(processed / total_files)
            
            # Display summary only
            status_text.text("✅ ICA processing completed!")
            st.success(f"Successfully processed {total_files} files and removed {total_artifacts_removed} artifact components total.")
            st.success("All files processed successfully!")
    
    elif step == "4. CNN-LSTM Training":
        st.header("Step 4: CNN-LSTM Model Training")
        
        if st.session_state.eeg_files is None or not hasattr(st.session_state, 'selected_indices'):
            st.warning("Please complete previous steps first!")
            return
        
        eeg_files = st.session_state.eeg_files
        selected_indices = st.session_state.selected_indices
        
        # Model parameters
        st.subheader("Model Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            desired_timesteps = st.number_input("Desired timesteps:", value=90000, min_value=1000, step=1000)
            epochs = st.number_input("Training epochs:", value=10, min_value=1, max_value=100)
            batch_size = st.selectbox("Batch size:", [2, 4, 8, 16], index=1)
        
        with col2:
            lstm_units = st.number_input("LSTM units:", value=64, min_value=8, max_value=512, step=8)
            dense_units = st.number_input("Dense units:", value=64, min_value=8, max_value=512, step=8)
            dropout_rate = st.slider("Dropout rate:", 0.0, 0.8, 0.3, 0.1)
        
        # LSTM Advanced Configuration
        st.subheader("LSTM Advanced Configuration")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            num_lstm_layers = st.number_input("Number of LSTM layers:", value=1, min_value=1, max_value=3)
            lstm_dropout = st.slider("LSTM dropout (between layers):", 0.0, 0.8, 0.0, 0.1)
        
        with col2:
            bidirectional = st.checkbox("Bidirectional LSTM", value=False)
            lstm_bias = st.checkbox("LSTM bias", value=True)
        
        with col3:
            if bidirectional:
                st.info("Bidirectional: Processes sequence forward and backward")
            if num_lstm_layers > 1:
                st.info(f"Stacking {num_lstm_layers} LSTM layers")
        
        # CNN Configuration
        st.subheader("CNN Layer Configuration")
        num_cnn_layers = st.selectbox("Number of CNN layers:", [1, 2, 3], index=1)
        
        cnn_config = []
        for i in range(num_cnn_layers):
            st.write(f"**CNN Layer {i+1}:**")
            col1, col2, col3 = st.columns(3)
            filters = col1.number_input(f"Filters {i+1}:", value=32*(2**i), min_value=8, max_value=256, step=8, key=f"filters_{i}")
            kernel_size = col2.number_input(f"Kernel size {i+1}:", value=5, min_value=3, max_value=15, step=2, key=f"kernel_{i}")
            pool_size = col3.number_input(f"Pool size {i+1}:", value=2, min_value=2, max_value=5, key=f"pool_{i}")
            cnn_config.append({'filters': filters, 'kernel_size': kernel_size, 'pool_size': pool_size})
        
        # Data Split Configuration
        st.subheader("📊 Data Split Configuration")
        st.write("Configure how to split your data into training, validation, and test sets:")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            train_ratio = st.slider("Training Set %", min_value=40, max_value=80, value=60, step=5)
        with col2:
            val_ratio = st.slider("Validation Set %", min_value=10, max_value=40, value=20, step=5)
        with col3:
            test_ratio = st.slider("Test Set %", min_value=10, max_value=40, value=20, step=5)
        
        # Validate ratios sum to 100%
        total_ratio = train_ratio + val_ratio + test_ratio
        if total_ratio != 100:
            st.error(f"⚠️ Data split ratios must sum to 100%. Current total: {total_ratio}%")
            st.info("Please adjust the sliders so the three percentages add up to exactly 100%.")
        else:
            st.success(f"✅ Data split: {train_ratio}% training, {val_ratio}% validation, {test_ratio}% test")
        
        # Disable training button if ratios don't sum to 100%
        training_enabled = (total_ratio == 100)
        
        if st.button("Start Training", disabled=not training_enabled):
            with st.spinner("Preparing data and training model..."):
                try:
                    # Prepare data
                    X, y, class_names = prepare_cnn_lstm_data(eeg_files, selected_indices, desired_timesteps)
                    num_classes = len(class_names)
                    
                    st.write(f"**Data prepared:**")
                    st.write(f"- Shape: {X.shape}")
                    st.write(f"- Classes: {class_names}")
                    st.write(f"- Samples per class: {np.bincount(y)}")
                    
                    # Convert to PyTorch tensors
                    X_tensor = torch.FloatTensor(X)
                    y_tensor = torch.LongTensor(y)
                    
                    # Split data into train, validation, and test sets using user-defined ratios
                    from sklearn.model_selection import train_test_split
                    
                    # Convert percentages to decimals
                    test_size = test_ratio / 100.0
                    val_size = val_ratio / 100.0
                    train_size = train_ratio / 100.0
                    
                    # First split: separate test set
                    X_temp, X_test, y_temp, y_test = train_test_split(
                        X_tensor, y_tensor, test_size=test_size, random_state=42, stratify=y
                    )
                    
                    # Second split: divide remaining into train and validation
                    # Calculate validation ratio from remaining data after test split
                    remaining_ratio = 1.0 - test_size  # e.g., 0.8 if test is 20%
                    val_from_remaining = val_size / remaining_ratio  # e.g., 0.2/0.8 = 0.25
                    
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_temp, y_temp, test_size=val_from_remaining, random_state=42, stratify=y_temp.numpy()
                    )
                    
                    # Build model
                    model = build_cnn_lstm_model(
                        input_shape=(X.shape[1], X.shape[2]),
                        num_classes=num_classes,
                        cnn_config=cnn_config,
                        lstm_units=lstm_units,
                        dense_units=dense_units,
                        dropout_rate=dropout_rate,
                        num_lstm_layers=num_lstm_layers,
                        lstm_dropout=lstm_dropout,
                        bidirectional=bidirectional,
                        lstm_bias=lstm_bias
                    )
                    
                    # Display model info
                    st.subheader("Model Architecture")
                    
                    # Display data split information
                    st.subheader("📊 Data Split Information")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        actual_train_pct = len(X_train)/len(X_tensor)*100
                        st.metric("Training Set", f"{len(X_train)} samples", f"{actual_train_pct:.1f}% (target: {train_ratio}%)")
                    with col2:
                        actual_val_pct = len(X_val)/len(X_tensor)*100
                        st.metric("Validation Set", f"{len(X_val)} samples", f"{actual_val_pct:.1f}% (target: {val_ratio}%)")
                    with col3:
                        actual_test_pct = len(X_test)/len(X_tensor)*100
                        st.metric("Test Set", f"{len(X_test)} samples", f"{actual_test_pct:.1f}% (target: {test_ratio}%)")
                    
                    st.write(f"Model: {model}")
                    
                    # Setup training
                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.Adam(model.parameters(), lr=0.001)
                    
                    # Create data loaders
                    from torch.utils.data import TensorDataset, DataLoader
                    train_dataset = TensorDataset(X_train, y_train)
                    val_dataset = TensorDataset(X_val, y_val)
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                    
                    # Training loop
                    st.subheader("Training Progress")
                    progress_bar = st.progress(0)
                    
                    # Create empty containers for updates
                    metrics_placeholder = st.empty()
                    graph_placeholder = st.empty()
                    
                    train_losses = []
                    val_losses = []
                    train_accuracies = []
                    val_accuracies = []
                    
                    for epoch in range(epochs):
                        # Training phase
                        model.train()
                        train_loss = 0.0
                        train_correct = 0
                        train_total = 0
                        
                        for batch_X, batch_y in train_loader:
                            optimizer.zero_grad()
                            outputs = model(batch_X)
                            loss = criterion(outputs, batch_y)
                            loss.backward()
                            optimizer.step()
                            
                            train_loss += loss.item()
                            _, predicted = torch.max(outputs.data, 1)
                            train_total += batch_y.size(0)
                            train_correct += (predicted == batch_y).sum().item()
                        
                        # Validation phase
                        model.eval()
                        val_loss = 0.0
                        val_correct = 0
                        val_total = 0
                        
                        with torch.no_grad():
                            for batch_X, batch_y in val_loader:
                                outputs = model(batch_X)
                                loss = criterion(outputs, batch_y)
                                
                                val_loss += loss.item()
                                _, predicted = torch.max(outputs.data, 1)
                                val_total += batch_y.size(0)
                                val_correct += (predicted == batch_y).sum().item()
                        
                        # Calculate metrics
                        epoch_train_loss = train_loss / len(train_loader)
                        epoch_val_loss = val_loss / len(val_loader)
                        epoch_train_acc = train_correct / train_total
                        epoch_val_acc = val_correct / val_total
                        
                        train_losses.append(epoch_train_loss)
                        val_losses.append(epoch_val_loss)
                        train_accuracies.append(epoch_train_acc)
                        val_accuracies.append(epoch_val_acc)
                        
                        # Update progress
                        progress_bar.progress((epoch + 1) / epochs)
                        
                        # Update metrics display (replace previous content)
                        with metrics_placeholder.container():
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Current Epoch", f"{epoch + 1}/{epochs}")
                                progress_pct = ((epoch + 1) / epochs) * 100
                                st.metric("Progress", f"{progress_pct:.1f}%")
                            with col2:
                                st.metric("Train Accuracy", f"{epoch_train_acc:.4f}", f"{epoch_train_acc*100:.1f}%")
                                st.metric("Train Loss", f"{epoch_train_loss:.4f}")
                            with col3:
                                st.metric("Val Accuracy", f"{epoch_val_acc:.4f}", f"{epoch_val_acc*100:.1f}%")
                                st.metric("Val Loss", f"{epoch_val_loss:.4f}")
                        
                        # Update real-time training progress graphs (replace previous graph)
                        if epoch >= 0:  # Show graphs from first epoch
                            with graph_placeholder.container():
                                st.markdown("### 📈 Real-time Training Progress")
                                
                                # Create side-by-side plots
                                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                                
                                # Accuracy plot
                                epochs_range = list(range(1, epoch + 2))
                                ax1.plot(epochs_range, train_accuracies, 'b-o', label='Training Accuracy', linewidth=2, markersize=6)
                                ax1.plot(epochs_range, val_accuracies, 'r-o', label='Validation Accuracy', linewidth=2, markersize=6)
                                ax1.set_title('Model Accuracy Progress', fontsize=14, fontweight='bold')
                                ax1.set_ylabel('Accuracy', fontsize=12)
                                ax1.set_xlabel('Epoch', fontsize=12)
                                ax1.legend(fontsize=11)
                                ax1.grid(True, alpha=0.3)
                                ax1.set_ylim(0, 1)
                                
                                # Loss plot
                                ax2.plot(epochs_range, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=6)
                                ax2.plot(epochs_range, val_losses, 'r-o', label='Validation Loss', linewidth=2, markersize=6)
                                ax2.set_title('Model Loss Progress', fontsize=14, fontweight='bold')
                                ax2.set_ylabel('Loss', fontsize=12)
                                ax2.set_xlabel('Epoch', fontsize=12)
                                ax2.legend(fontsize=11)
                                ax2.grid(True, alpha=0.3)
                                
                                # Add current values as annotations on the latest points
                                if len(epochs_range) > 0:
                                    ax1.annotate(f'{epoch_train_acc:.3f}', 
                                               (epoch + 1, epoch_train_acc), 
                                               textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, 
                                               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
                                    ax1.annotate(f'{epoch_val_acc:.3f}', 
                                               (epoch + 1, epoch_val_acc), 
                                               textcoords="offset points", xytext=(0,10), ha='center', fontsize=9,
                                               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close()  # Close to prevent memory issues
                    
                    # Store results in session state
                    st.session_state.model = model
                    st.session_state.train_losses = train_losses
                    st.session_state.val_losses = val_losses
                    st.session_state.train_accuracies = train_accuracies
                    st.session_state.val_accuracies = val_accuracies
                    st.session_state.class_names = class_names
                    
                    # Save best model based on validation accuracy
                    best_val_epoch = np.argmax(val_accuracies)
                    best_val_acc = val_accuracies[best_val_epoch]
                    
                    # Store model configuration for saving/loading
                    model_config = {
                        'input_shape': (X.shape[1], X.shape[2]),
                        'num_classes': num_classes,
                        'cnn_config': cnn_config,
                        'lstm_units': lstm_units,
                        'dense_units': dense_units,
                        'dropout_rate': dropout_rate,
                        'num_lstm_layers': num_lstm_layers,
                        'lstm_dropout': lstm_dropout,
                        'bidirectional': bidirectional,
                        'lstm_bias': lstm_bias,
                        'class_names': class_names,
                        'selected_indices': selected_indices,
                        'best_epoch': best_val_epoch + 1,
                        'best_val_accuracy': best_val_acc,
                        'final_train_accuracy': train_accuracies[-1],
                        'final_val_accuracy': val_accuracies[-1]
                    }
                    
                    st.session_state.model_config = model_config
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test
                    
                    st.success("✅ PyTorch CNN-LSTM model training completed!")
                    
                    # Training completion summary
                    st.subheader("🎯 Training Completed Successfully!")
                    
                    # Display final training summary in a clean format
                    st.info(f"""
                    **Training Summary:**
                    - Total Epochs: {epochs}
                    - Final Training Accuracy: {train_accuracies[-1]:.4f} ({train_accuracies[-1]*100:.1f}%)
                    - Final Validation Accuracy: {val_accuracies[-1]:.4f} ({val_accuracies[-1]*100:.1f}%)
                    - Training Loss: {train_losses[-1]:.4f}
                    - Validation Loss: {val_losses[-1]:.4f}
                    """)
                    
                    # Final results metrics
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("🎯 Final Train Accuracy", f"{train_accuracies[-1]:.4f}", f"{train_accuracies[-1]*100:.1f}%")
                    col2.metric("📊 Final Val Accuracy", f"{val_accuracies[-1]:.4f}", f"{val_accuracies[-1]*100:.1f}%")
                    col3.metric("📉 Final Train Loss", f"{train_losses[-1]:.4f}")
                    col4.metric("📈 Final Val Loss", f"{val_losses[-1]:.4f}")
                    
                    # Final comprehensive training history plot
                    st.subheader("📈 Complete Training History")
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                    
                    # Accuracy plot
                    epochs_range = list(range(1, epochs + 1))
                    ax1.plot(epochs_range, train_accuracies, 'b-o', label='Training Accuracy', linewidth=2, markersize=4)
                    ax1.plot(epochs_range, val_accuracies, 'r-o', label='Validation Accuracy', linewidth=2, markersize=4)
                    ax1.set_title('Model Accuracy Over Time', fontsize=14, fontweight='bold')
                    ax1.set_ylabel('Accuracy', fontsize=12)
                    ax1.set_xlabel('Epoch', fontsize=12)
                    ax1.legend(fontsize=11)
                    ax1.grid(True, alpha=0.3)
                    ax1.set_ylim(0, 1)
                    
                    # Loss plot
                    ax2.plot(epochs_range, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=4)
                    ax2.plot(epochs_range, val_losses, 'r-o', label='Validation Loss', linewidth=2, markersize=4)
                    ax2.set_title('Model Loss Over Time', fontsize=14, fontweight='bold')
                    ax2.set_ylabel('Loss', fontsize=12)
                    ax2.set_xlabel('Epoch', fontsize=12)
                    ax2.legend(fontsize=11)
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()  # Close to prevent memory issues
                    
                    # Final Test Evaluation
                    st.subheader("🧪 Final Test Evaluation")
                    st.info("Evaluating model on unseen test data...")
                    
                    model.eval()
                    with torch.no_grad():
                        test_outputs = model(X_test)
                        test_loss = criterion(test_outputs, y_test)
                        _, test_predicted = torch.max(test_outputs.data, 1)
                        test_accuracy = (test_predicted == y_test).sum().item() / len(y_test)
                    
                    # Display test results
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("🎯 Final Test Accuracy", f"{test_accuracy:.4f}", f"{test_accuracy:.2%}")
                    with col2:
                        st.metric("📊 Final Train Accuracy", f"{train_accuracies[-1]:.4f}")
                    with col3:
                        st.metric("📈 Final Validation Accuracy", f"{val_accuracies[-1]:.4f}")
                    with col4:
                        st.metric("🔍 Test Loss", f"{test_loss:.4f}")
                    
                    # Performance comparison
                    st.subheader("📈 Performance Summary")
                    performance_data = {
                        'Dataset': ['Training', 'Validation', 'Test'],
                        'Accuracy': [train_accuracies[-1], val_accuracies[-1], test_accuracy],
                        'Sample Count': [len(X_train), len(X_val), len(X_test)]
                    }
                    df_performance = pd.DataFrame(performance_data)
                    st.dataframe(df_performance, use_container_width=True)
                    
                    # Check for overfitting
                    train_val_gap = train_accuracies[-1] - val_accuracies[-1]
                    val_test_gap = val_accuracies[-1] - test_accuracy
                    
                    if train_val_gap > 0.1:
                        st.warning(f"⚠️ Potential overfitting detected: Training accuracy ({train_accuracies[-1]:.3f}) is significantly higher than validation accuracy ({val_accuracies[-1]:.3f})")
                    
                    if abs(val_test_gap) < 0.05:
                        st.success("✅ Good generalization: Validation and test accuracies are very similar!")
                    elif val_test_gap > 0.05:
                        st.warning("⚠️ Model may have overfit to validation set")
                    else:
                        st.info("📊 Test performance is better than validation - good sign!")
                    
                except Exception as e:
                    st.error(f"Error during training: {e}")
                    import traceback
                    st.text(traceback.format_exc())
    
    elif step == "5. Model Testing":
        st.header("Step 5: Model Testing & Management")
        
        if not hasattr(st.session_state, 'model') or st.session_state.model is None:
            st.warning("Please train a model first!")
            return
        
        # Model management section
        st.subheader("💾 Model Management")
        
        if hasattr(st.session_state, 'model_config'):
            config = st.session_state.model_config
            
            # Display model info
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"""
                **Current Model Info:**
                - Input Shape: {config['input_shape']}
                - Classes: {len(config['class_names'])}
                - Best Epoch: {config['best_epoch']}
                - Best Val Accuracy: {config['best_val_accuracy']:.4f}
                """)
            
            with col2:
                st.info(f"""
                **Architecture:**
                - LSTM Units: {config['lstm_units']}
                - LSTM Layers: {config['num_lstm_layers']}
                - Bidirectional: {config['bidirectional']}
                - Dense Units: {config['dense_units']}
                """)
            
            # Save model
            col1, col2 = st.columns(2)
            with col1:
                model_name = st.text_input("Model Name", value="cnn_lstm_model")
                
                if st.button("💾 Save Model"):
                    try:
                        # Create models directory if it doesn't exist
                        os.makedirs("saved_models", exist_ok=True)
                        
                        # Save model state dict
                        model_path = f"saved_models/{model_name}.pth"
                        torch.save(st.session_state.model.state_dict(), model_path)
                        
                        # Save model configuration
                        config_path = f"saved_models/{model_name}_config.pkl"
                        import pickle
                        with open(config_path, 'wb') as f:
                            pickle.dump(config, f)
                        
                        st.success(f"✅ Model saved successfully!")
                        st.info(f"Model files:\n- {model_path}\n- {config_path}")
                        
                    except Exception as e:
                        st.error(f"Error saving model: {e}")
            
            with col2:
                # Load model
                st.write("**Load Saved Model:**")
                saved_models = []
                if os.path.exists("saved_models"):
                    saved_models = [f.replace('.pth', '') for f in os.listdir("saved_models") if f.endswith('.pth')]
                
                if saved_models:
                    selected_model = st.selectbox("Select saved model:", saved_models)
                    
                    if st.button("📂 Load Model"):
                        try:
                            # Load configuration
                            config_path = f"saved_models/{selected_model}_config.pkl"
                            import pickle
                            with open(config_path, 'rb') as f:
                                loaded_config = pickle.load(f)
                            
                            # Rebuild model with loaded config
                            loaded_model = build_cnn_lstm_model(
                                input_shape=loaded_config['input_shape'],
                                num_classes=loaded_config['num_classes'],
                                cnn_config=loaded_config['cnn_config'],
                                lstm_units=loaded_config['lstm_units'],
                                dense_units=loaded_config['dense_units'],
                                dropout_rate=loaded_config['dropout_rate'],
                                num_lstm_layers=loaded_config['num_lstm_layers'],
                                lstm_dropout=loaded_config['lstm_dropout'],
                                bidirectional=loaded_config['bidirectional'],
                                lstm_bias=loaded_config['lstm_bias']
                            )
                            
                            # Load model weights
                            model_path = f"saved_models/{selected_model}.pth"
                            loaded_model.load_state_dict(torch.load(model_path))
                            
                            # Update session state
                            st.session_state.model = loaded_model
                            st.session_state.model_config = loaded_config
                            
                            st.success(f"✅ Model '{selected_model}' loaded successfully!")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error loading model: {e}")
                else:
                    st.info("No saved models found")
        
        st.markdown("---")
        
        # Test on unseen data section
        st.subheader("🧪 Test on Unseen Data")
        
        # Option 1: Test on holdout test set
        if hasattr(st.session_state, 'X_test') and hasattr(st.session_state, 'y_test'):
            st.write("**Option 1: Test on Holdout Test Set**")
            
            if st.button("🎯 Evaluate on Test Set"):
                try:
                    model = st.session_state.model
                    X_test = st.session_state.X_test
                    y_test = st.session_state.y_test
                    class_names = st.session_state.model_config['class_names']
                    
                    model.eval()
                    with torch.no_grad():
                        test_outputs = model(X_test)
                        _, test_predicted = torch.max(test_outputs, 1)
                        test_accuracy = (test_predicted == y_test).sum().item() / len(y_test)
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    col1.metric("🎯 Test Accuracy", f"{test_accuracy:.4f}", f"{test_accuracy*100:.1f}%")
                    col2.metric("📊 Test Samples", len(y_test))
                    col3.metric("🎲 Correct Predictions", (test_predicted == y_test).sum().item())
                    
                    # Confusion matrix
                    from sklearn.metrics import confusion_matrix, classification_report
                    cm = confusion_matrix(y_test.cpu().numpy(), test_predicted.cpu().numpy())
                    
                    st.write("**Confusion Matrix:**")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                               xticklabels=class_names, yticklabels=class_names, ax=ax)
                    ax.set_title('Confusion Matrix')
                    ax.set_ylabel('True Label')
                    ax.set_xlabel('Predicted Label')
                    st.pyplot(fig)
                    plt.close()
                    
                    # Classification report
                    st.write("**Classification Report:**")
                    report = classification_report(y_test.cpu().numpy(), test_predicted.cpu().numpy(), 
                                                 target_names=class_names, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df)
                    
                except Exception as e:
                    st.error(f"Error during testing: {e}")
        
        st.markdown("---")
        
        # Option 2: Upload new data for testing
        st.write("**Option 2: Upload New Data for Testing**")
        
        uploaded_test_file = st.file_uploader("Upload new EEG data (ZIP format)", type="zip", key="test_data")
        
        if uploaded_test_file is not None:
            with st.spinner("Processing new test data..."):
                try:
                    # Extract uploaded test data
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
                        tmp_file.write(uploaded_test_file.read())
                        tmp_file_path = tmp_file.name
                    
                    test_eeg_files = extract_zip(tmp_file_path)
                    os.unlink(tmp_file_path)  # Clean up
                    
                    if test_eeg_files:
                        st.success(f"✅ Extracted {sum(len(files) for files in test_eeg_files.values())} test files")
                        
                        # Use the same selected indices from training
                        if hasattr(st.session_state, 'model_config'):
                            selected_indices = st.session_state.model_config['selected_indices']
                            class_names = st.session_state.model_config['class_names']
                            
                            if st.button("🚀 Test on New Data"):
                                # Prepare test data using same preprocessing
                                X_new, y_new, _ = prepare_cnn_lstm_data(test_eeg_files, selected_indices, 90000)
                                
                                # Convert to tensors
                                X_new_tensor = torch.FloatTensor(X_new)
                                y_new_tensor = torch.LongTensor(y_new)
                                
                                # Test the model
                                model = st.session_state.model
                                model.eval()
                                with torch.no_grad():
                                    outputs = model(X_new_tensor)
                                    _, predicted = torch.max(outputs, 1)
                                    accuracy = (predicted == y_new_tensor).sum().item() / len(y_new_tensor)
                                
                                # Display results
                                st.subheader("📊 New Data Test Results")
                                col1, col2, col3 = st.columns(3)
                                col1.metric("🎯 Accuracy", f"{accuracy:.4f}", f"{accuracy*100:.1f}%")
                                col2.metric("📊 Samples", len(y_new))
                                col3.metric("🎲 Correct", (predicted == y_new_tensor).sum().item())
                                
                                # Show predictions vs actual
                                results_df = pd.DataFrame({
                                    'Actual': [class_names[i] for i in y_new],
                                    'Predicted': [class_names[i] for i in predicted.cpu().numpy()],
                                    'Correct': (predicted == y_new_tensor).cpu().numpy()
                                })
                                
                                st.write("**Detailed Results:**")
                                st.dataframe(results_df)
                        else:
                            st.error("Model configuration not found. Please train a model first.")
                    else:
                        st.error("No valid EEG files found in the uploaded ZIP")
                        
                except Exception as e:
                    st.error(f"Error processing test data: {e}")

if __name__ == "__main__":
    main()