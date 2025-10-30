# =========================================
# data_loader.py
# =========================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

def load_data(csv_path="outputs/angles_pose_keypoints.csv"):
    df = pd.read_csv(csv_path)
    
    # Drop non-feature columns
    X = df.drop(["pose", "frame"], axis=1)
    y = df["pose"]
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Reshape for LSTM/GRU (sequence length = 1 for now)
    X = X.reshape(X.shape[0], 1, X.shape[1])
    
    # Split data
    return train_test_split(X, y, test_size=0.2, random_state=42), le
