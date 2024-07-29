import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR


def predict_X(lx, ly, lz):
    loaded_model = tf.keras.models.load_model('./data_from_backend/X/keras/X.keras')
    loaded_pca_mat = np.load('data_from_backend/X/pca/pca_mat_X.npy')
    loaded_pca_mean = np.load('data_from_backend/X/pca/pca_mean_X.npy')
    x_input = np.array([[lx, ly, lz]])
    y_pred = loaded_model.predict(x_input)
    y_pred = y_pred[0]
    y_pred_original = np.dot(y_pred, loaded_pca_mat) + loaded_pca_mean
    return y_pred_original.tolist()

def predict_Y(lx, ly, lz):
    loaded_model = tf.keras.models.load_model('./data_from_backend/Y/keras/Y.keras')
    loaded_pca_mat = np.load('data_from_backend/Y/pca/pca_mat_Y.npy')
    loaded_pca_mean = np.load('data_from_backend/Y/pca/pca_mean_Y.npy')
    x_input = np.array([[lx, ly, lz]])
    y_pred = loaded_model.predict(x_input)
    y_pred = y_pred[0]
    y_pred_original = np.dot(y_pred, loaded_pca_mat) + loaded_pca_mean
    return y_pred_original.tolist()

def predict_Z(lx, ly, lz):
    loaded_model = tf.keras.models.load_model('./data_from_backend/Z/keras/Z.keras')
    loaded_pca_mat = np.load('data_from_backend/Z/pca/pca_mat_Z.npy')
    loaded_pca_mean = np.load('data_from_backend/Z/pca/pca_mean_Z.npy')
    x_input = np.array([[lx, ly, lz]])
    y_pred = loaded_model.predict(x_input)
    y_pred = y_pred[0]
    y_pred_original = np.dot(y_pred, loaded_pca_mat) + loaded_pca_mean
    return y_pred_original.tolist()

def predict_force(lx, ly, lz):
    loaded_model = tf.keras.models.load_model('./data_from_backend/force/keras/Force.keras')
    loaded_pca_mat = np.load('data_from_backend/force/pca/pca_mat_force.npy')
    loaded_pca_mean = np.load('data_from_backend/force/pca/pca_mean_force.npy')
    x_input = np.array([[lx, ly, lz]])
    y_pred = loaded_model.predict(x_input)
    y_pred = y_pred[0]
    y_pred_original = np.dot(y_pred, loaded_pca_mat) + loaded_pca_mean
    return y_pred_original.tolist()

def predict_failure(lx, ly, lz):
    loaded_model = tf.keras.models.load_model('./data_from_backend/Failure/keras/Failure.keras')
    loaded_pca_mat = np.load('data_from_backend/Failure/pca/pca_mat_Failure.npy')
    loaded_pca_mean = np.load('data_from_backend/Failure/pca/pca_mean_Failure.npy')
    x_input = np.array([[lx, ly, lz]])
    y_pred = loaded_model.predict(x_input)
    y_pred = y_pred[0]
    y_pred_original = np.dot(y_pred, loaded_pca_mat) + loaded_pca_mean
    return y_pred_original.tolist()

def predict_Displacement(lx, ly, lz):
    input_dim = 3  
    output_dim = 10  

    class NeuralNetwork(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(NeuralNetwork, self).__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.2),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.2),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.2),
                nn.Linear(256, output_dim),
                nn.ReLU()
            )

        def forward(self, x):
            return self.layers(x) 

    model = NeuralNetwork(input_dim, output_dim)

    # Load the trained model parameters
    model.load_state_dict(torch.load('./data_from_backend/Displacement/keras/Displacement.pth'))
    model.eval()
    data = np.array([[lx, ly, lz]])
    with torch.no_grad():  # Ensure model is in evaluation mode for inference
        inputs = torch.tensor(data, dtype=torch.float32)
        predictions = model(inputs)
        loaded_pca_mat = np.load('data_from_backend/Displacement/pca/pca_mat_Displacement.npy')
        loaded_pca_mean = np.load('data_from_backend/Displacement/pca/pca_mean_Displacement.npy')
        predictions = predictions.numpy()
        predictions = np.dot(predictions, loaded_pca_mat) + loaded_pca_mean
        return predictions[0].tolist()  # Convert predictions to NumPy array if needed

def predict_maxForce(lx, ly, lz):
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.layers = nn.Sequential(
                nn.Linear(3, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.2),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.2),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.2),
                nn.Linear(128, 1),
            )

        def forward(self, x):
            return self.layers(x)
        
    model = NeuralNetwork()
    model.load_state_dict(torch.load('./data_from_backend/maxForce/keras/maxForce.pth'))
    model.eval()
    data = np.array([[lx, ly, lz]])
    with torch.no_grad():  # Ensure model is in evaluation mode for inference
        inputs = torch.tensor(data, dtype=torch.float32)
        predictions = model(inputs)
        return predictions[0].tolist()  # Convert predictions to NumPy array if needed

def predict_all_val(lx, ly, lz):
    nodal_defomation_x = predict_X(lx, ly, lz)
    nodal_defomation_Y = predict_Y(lx, ly, lz)
    nodal_defomation_Z = predict_Z(lx, ly, lz)
    nodal_failure = predict_failure(lx, ly, lz)
    force = predict_force(lx, ly, lz)
    displacement = predict_Displacement(lx, ly, lz)
    max_force = predict_maxForce(lx, ly, lz)

    return max_force, nodal_defomation_x, nodal_defomation_Y, nodal_defomation_Z, nodal_failure, displacement, force
