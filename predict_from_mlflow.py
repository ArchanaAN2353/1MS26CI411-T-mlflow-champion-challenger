
import mlflow
import mlflow.pyfunc
import pandas as pd
from pathlib import Path

# Connect to MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Champion model URI
model_uri = "models:/SentimentAnalysisClassifier@champion"

print("Loading Champion model from MLflow...")

# Load registered Champion model
model = mlflow.pyfunc.load_model(model_uri)

print("Champion model loaded successfully!")

# Read new reviews
data = pd.read_csv("data/new_data.csv")

# Generate predictions
predictions = model.predict(data[["text"]])

# Combine input and predictions
results = pd.concat(
    [data.reset_index(drop=True), predictions.reset_index(drop=True)],
    axis=1
)

# Save predictions
Path("outputs").mkdir(exist_ok=True)

results.to_csv(
    "outputs/champion_predictions.csv",
    index=False
)

print("\nCHAMPION MODEL PREDICTIONS")
print(results.to_string(index=False))

print("\nPredictions saved to outputs/champion_predictions.csv")
