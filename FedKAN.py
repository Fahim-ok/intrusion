import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin

# Federated Learning Setup
class FederatedLearning:
    def __init__(self, model, clients, epochs=10):
        self.model = model
        self.clients = clients
        self.epochs = epochs

    def train(self, X_train, y_train):
        client_data = np.array_split(X_train, self.clients)
        client_labels = np.array_split(y_train, self.clients)
        
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            client_models = []
            for i in range(self.clients):
                client_model = self.model
                client_model.fit(client_data[i], client_labels[i])
                client_models.append(client_model)
            
            # Aggregate model updates (simplified)
            aggregated_model = self.aggregate_models(client_models)
            self.model = aggregated_model

    def aggregate_models(self, models):
        # Simple averaging of client model weights (for illustration)
        avg_weights = np.mean([model.model.state_dict() for model in models], axis=0)
        self.model.model.load_state_dict(avg_weights)
        return self.model

# Apply TOPSIS to remove outliers based on accuracy scores
def apply_topsis(results):
    normalized_results = normalize(results)
    pca = PCA(n_components=1)
    transformed = pca.fit_transform(normalized_results)
    score = np.abs(transformed)  # Outliers have higher absolute scores
    return score

# Preprocessing function
def preprocess_data(data):
    X = data.drop(columns=['Attack Category', 'Label'])
    y = data['Attack Category']
    
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    X_preprocessed = preprocessor.fit_transform(X)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    return X_preprocessed, y_encoded, preprocessor, label_encoder

# Dummy class for a basic model (KanConvNet or any other can be used)
class DummyModel(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        # A mock fit function
        self.model = nn.Sequential(
            nn.Linear(X.shape[1], 50),
            nn.ReLU(),
            nn.Linear(50, len(np.unique(y)))
        )
        return self

    def predict(self, X):
        # A mock predict function
        return np.random.choice(np.unique(X), len(X))

# Example dataset loading
data = pd.read_csv('/kaggle/input/iomtdata/wustl-ehms-2020_with_attacks_categories.csv')  # Replace with your actual dataset

# Preprocess the data
X_preprocessed, y_encoded, preprocessor, label_encoder = preprocess_data(data)
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y_encoded, test_size=0.2, random_state=42)

# Federated Learning Setup and Training
federated_learning = FederatedLearning(DummyModel(), clients=5, epochs=5)
federated_learning.train(X_train, y_train)

# Predictions
y_pred = federated_learning.model.predict(X_test)

# Classification Report
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Apply TOPSIS (mock results based on accuracy scores for simplicity)
results = np.array([[accuracy_score(y_test, y_pred)]])
outlier_scores = apply_topsis(results)
print(f"Outlier Scores: {outlier_scores}")
