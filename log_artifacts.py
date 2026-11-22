import mlflow
from pathlib import Path

# Connect to MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Find the existing experiment
experiment = mlflow.get_experiment_by_name(
    "Sentiment Analysis Champion Challenger"
)

# Start a new MLflow run
with mlflow.start_run(
    experiment_id=experiment.experiment_id,
    run_name="final_project_artifacts"
):

    # Log graphs
    mlflow.log_artifact("outputs/performance_comparison.png")
    mlflow.log_artifact("outputs/latency_comparison.png")

    # Log project outputs
    mlflow.log_artifact("outputs/metrics_summary.csv")
    mlflow.log_artifact("outputs/decision.json")
    mlflow.log_artifact("outputs/champion_predictions.csv")

    print("All project artifacts logged successfully!")

print("Check the MLflow dashboard.")