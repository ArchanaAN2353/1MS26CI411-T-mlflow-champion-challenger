
import os
import time
import json
import pandas as pd
import mlflow
import mlflow.pyfunc
from transformers import pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# MLflow setup
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Sentiment Analysis Champion Challenger")

# Load dataset
data = pd.read_csv("data/eval_inputs.csv")
texts = data["text"].astype(str).tolist()
labels = data["label"].astype(int).tolist()

# Models to compare
models = {
    "A": "distilbert-base-uncased-finetuned-sst-2-english",
    "B": "textattack/bert-base-uncased-SST-2"
}

results = []

for role, model_id in models.items():

    print(f"\nEvaluating Model {role}: {model_id}")

    classifier = pipeline(
        "sentiment-analysis",
        model=model_id,
        device=-1
    )

    start = time.time()
    predictions = classifier(texts, batch_size=8)
    total_time = time.time() - start

    predicted_labels = [
        1 if p["label"] in ["POSITIVE", "LABEL_1"] else 0
        for p in predictions
    ]

    metrics = {
        "accuracy": accuracy_score(labels, predicted_labels),
        "precision": precision_score(labels, predicted_labels, zero_division=0),
        "recall": recall_score(labels, predicted_labels, zero_division=0),
        "f1_score": f1_score(labels, predicted_labels, zero_division=0),
        "avg_latency_ms": (total_time / len(texts)) * 1000,
        "total_inference_s": total_time
    }

    results.append({
        "model": role,
        "hf_model_id": model_id,
        **metrics
    })

    predictions_df = pd.DataFrame({
        "text": texts,
        "actual_label": labels,
        "predicted_label": predicted_labels,
        "confidence": [p["score"] for p in predictions],
        "correct": [
            actual == predicted
            for actual, predicted in zip(labels, predicted_labels)
        ]
    })

    predictions_df.to_csv(
        f"outputs/predictions_model_{role}.csv",
        index=False
    )

    with mlflow.start_run(run_name=f"{role}_sentiment_{role}"):

        mlflow.log_param("hf_model_id", model_id)
        mlflow.log_param("task", "sentiment-analysis")
        mlflow.log_param("role", role)
        mlflow.log_param("eval_samples", len(texts))
        mlflow.log_param("device", "CPU")

        mlflow.log_metrics(metrics)

        mlflow.set_tag("dataset_name", "custom_sentiment_dataset")
        mlflow.set_tag("student_usn", "1MS26CI411-T")

        mlflow.log_artifact(
            f"outputs/predictions_model_{role}.csv"
        )

    print("Metrics:", metrics)

# Compare models
summary = pd.DataFrame(results)
summary.to_csv("outputs/metrics_summary.csv", index=False)

# Decision rule: higher F1 is better.
# If F1 scores are within 0.01, choose the faster model.
if abs(summary.loc[0, "f1_score"] - summary.loc[1, "f1_score"]) <= 0.01:
    winner = summary.loc[
        summary["avg_latency_ms"].idxmin(), "model"
    ]
else:
    winner = summary.loc[
        summary["f1_score"].idxmax(), "model"
    ]

with mlflow.start_run(run_name="compare_sentiment"):

    mlflow.log_param("primary_metric", "f1_score")
    mlflow.log_param("minimum_improvement_margin", 0.01)
    mlflow.log_metric(
        "f1_delta",
        abs(summary.loc[0, "f1_score"] - summary.loc[1, "f1_score"])
    )
    mlflow.set_tag("winner", winner)

print("\nMODEL COMPARISON")
print(summary.to_string(index=False))
print("\nChampion Model:", winner)

with open("outputs/decision.json", "w") as file:
    json.dump({
        "champion": winner,
        "challenger": "B" if winner == "A" else "A",
        "decision_rule": "Higher F1; if within 0.01, lower latency wins."
    }, file, indent=4)
