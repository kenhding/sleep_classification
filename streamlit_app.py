import streamlit as st
import mne
import os
from pathlib import Path
import subprocess
import hashlib  # Add this import at the beginning
import numpy as np
import plotly.graph_objects as go  # Add this import for Plotly

# Define the project folder path (relative to where the Streamlit app is run)
PROJECT_DIR = Path(__file__).parent  # This will get the current directory where the script is located

# Define the relative directories for uploading files and output
UPLOAD_DIR = PROJECT_DIR / 'uploaded_files'  # Folder for storing uploaded files
OUTPUT_DIR = PROJECT_DIR / 'processed_files'  # Folder for storing processed files
PREDICTION_DIR = PROJECT_DIR / 'predictions'  # Folder for storing predictions

# Ensure that the directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)  # Create the upload folder if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # Create the output folder if it doesn't exist
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)  # Create the prediction folder if it doesn't exist

# Function to handle the file upload and processing
def process_edf_file(uploaded_file):
    # Step 1: Rename the uploaded file to lowercase and replace spaces with underscores
    file_name_lowercase = uploaded_file.name.lower().replace(" ", "_")
    hashed_name = hashlib.md5(file_name_lowercase.encode()).hexdigest()  # MD5 produces a 32-character hash
    new_file_name = f"{hashed_name}.edf"  # Append the .edf extension
        
    # Save the uploaded file with the new name to the UPLOAD_DIR
    file_path = UPLOAD_DIR / new_file_name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        # Read the EDF file with light loading (metadata only)
        raw = mne.io.read_raw_edf(file_path, preload=False)
        
        # Get the channel names
        channel_names = raw.info['ch_names']
        
        # Additional information (sampling frequency, etc.)
        st.write(f"Sampling frequency: {raw.info['sfreq']} Hz")
        st.write(f"Number of channels: {len(channel_names)}")
            
        # Display the channel names
        st.write(f"Channel names for {uploaded_file.name}:")

        # Let the user select the EEG and EOG channels from the list of available channels
        eeg_channel = st.selectbox("Select the EEG channel", options=[None] + channel_names)  # Default is None
        eog_channel = st.selectbox("Select the EOG channel", options=[None] + channel_names)  # Default is None

        if eeg_channel is not None and eog_channel is not None:
            # Display the selected EEG and EOG channels
            st.write(f"Selected EEG channel: {eeg_channel}")
            st.write(f"Selected EOG channel: {eog_channel}")

            # 1. Run the external command to preprocess the data using 'ut extract'
            preprocess_command = [
                'ut', 'extract', '--overwrite', '--file_regex', f'{UPLOAD_DIR}/*.edf',
                '--out_dir', str(OUTPUT_DIR), '--resample', '128',
                '--channels', eeg_channel, eog_channel,
                '--rename_channels', 'EEG1', 'EOG1'  # You may want to rename the channels here if needed
            ]
            
            try:
                # Run the preprocessing command
                subprocess.run(preprocess_command, check=True)
                st.write("Preprocessing command executed successfully!")

                # 2. Find the corresponding preprocessed .h5 file
                base_name = Path(new_file_name).stem  # Remove the file extension from the uploaded file name
                preprocessed_file = OUTPUT_DIR / base_name / f"{base_name}.h5"
                st.write(preprocessed_file)
                
                # Check if the preprocessed .h5 file exists
                if preprocessed_file.exists():
                    st.write(f"Preprocessed file found: {preprocessed_file}")
                    
                    # 3. Run the external command for prediction using 'ut predict_one'
                    predict_command = [
                        'ut', 'predict_one', '-f', str(preprocessed_file), '-o', str(PREDICTION_DIR),
                        '--channels', 'EEG1==EOG', 'EOG1==EOG',
                        '--overwrite'
                    ]
                    # Run the prediction command
                    subprocess.run(predict_command, check=True)
                    st.write("Prediction command executed successfully!")

                    # 4. Plot the prediction result (Assuming the result is in .npy format)
                    npy_file_path = PREDICTION_DIR / f"{base_name}.npy"
                    if npy_file_path.exists():
                        # Load the prediction data
                        prediction_data = np.load(npy_file_path).ravel()

                        # Display the shape of the prediction data
                        st.write(f"Prediction data shape: {prediction_data.shape}")

                        # Calculate the time axis (each point is 30 seconds apart)
                        time_axis = np.arange(len(prediction_data)) * 30  # Time in seconds

                        # Plot the prediction data using Plotly
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=time_axis, y=prediction_data, mode='lines', name='Prediction'))

                        fig.update_layout(
                            title='Prediction Output Over Time',
                            xaxis_title='Time (seconds)',
                            yaxis_title='Prediction Value'
                        )

                        st.plotly_chart(fig)
                    else:
                        st.write("Error: Prediction .npy file not found.")

                else:
                    st.write("Error: Preprocessed file (.h5) not found.")
                
            except subprocess.CalledProcessError as e:
                st.write(f"Error running command: {e}")

    except Exception as e:
        st.write(f"Error reading {uploaded_file.name}: {e}")

# Upload a single EDF file
uploaded_file = st.file_uploader("Choose an EDF file", type=["edf"])

# Process the uploaded file if one is selected
if uploaded_file is not None:
    process_edf_file(uploaded_file)
