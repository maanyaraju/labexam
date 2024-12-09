import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import load_diabetes, load_breast_cancer
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

# Load Sonar Dataset
def load_sonar():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
    data = pd.read_csv(url, header=None)
    X = data.iloc[:, :-1].values
    y = LabelEncoder().fit_transform(data.iloc[:, -1].values)
    return X, y

# Load datasets
datasets = {
    "Diabetes": load_diabetes(return_X_y=True),
    "Cancer": load_breast_cancer(return_X_y=True),
    "Sonar": load_sonar()
}

# Define the Neural Network model
def build_model(input_dim, output_dim, task_type="classification"):
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_dim=input_dim),
        layers.Dense(32, activation='relu')
    ])
    
    if task_type == "classification" and output_dim > 1:  # Multi-class
        model.add(layers.Dense(output_dim, activation='softmax'))
        loss = 'sparse_categorical_crossentropy'
    elif task_type == "classification":  # Binary classification
        model.add(layers.Dense(1, activation='sigmoid'))
        loss = 'binary_crossentropy'
    else:  # Regression
        model.add(layers.Dense(1))
        loss = 'mean_squared_error'

    model.compile(optimizer='adam', loss=loss, metrics=['accuracy' if task_type == "classification" else 'mse'])
    return model

# Train and evaluate the model
results = {}
for dataset_name, (X, y) in datasets.items():
    print(f"\nTraining on {dataset_name} dataset")
    
    # Determine task type
    task_type = "classification" if dataset_name != "Diabetes" else "regression"
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Define the model
    output_dim = len(np.unique(y)) if task_type == "classification" else 1
    model = build_model(X_train.shape[1], output_dim, task_type)
    
    # Train the model
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
    
    # Evaluate the model
    loss, metric = model.evaluate(X_test, y_test, verbose=0)
    metric_name = 'accuracy' if task_type == "classification" else 'mse'
    results[dataset_name] = {"loss": loss, metric_name: metric}
    print(f"{dataset_name} - Loss: {loss:.4f}, {metric_name.capitalize()}: {metric:.4f}")

# Display final results
print("\nFinal Results:")
for dataset, metrics in results.items():
    for key, value in metrics.items():
        print(f"{dataset} - {key.capitalize()}: {value:.4f}")
