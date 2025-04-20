import numpy as np
from sklearn.preprocessing import MinMaxScaler

def scale_input(df):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['x', 'y']])
    return scaled_data, scaler

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x_seq = data[i:i+seq_length]
        y_seq = data[i+seq_length]
        xs.append(x_seq)
        ys.append(y_seq)
    return np.array(xs), np.array(ys)
