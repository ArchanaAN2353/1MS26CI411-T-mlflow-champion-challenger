
import mlflow
import mlflow.pyfunc
import pandas as pd
from transformers import pipeline

# MLflow setup
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_registry_uri("http://127.0.0.1:5000")

MODEL_NAME = "SentimentAnalysisClassifier"


# Custom sentiment model
class SentimentModel(mlflow.pyfunc.PythonModel):

    def __init__(self, model_id):
        self.model_id = model_id
        self.classifier = None

    def load_context(self, context):
        self.classifier = pipeline(
            "sentiment-analysis",
            model=self.model_id,
            device=-1
        )

    def predict(self, context, model_input, params=None):

        texts = model_input["text"].astype(str).tolist()

        results = self.classifier(texts, batch_size=8)

        return pd.DataFrame({
            "sentiment": [r["label"] for r in results],
            "confidence": [r["score"] for r in results]
        })


# Models to register
models = {
    "A": "distilbert-base-uncased-finetuned-sst-2-english",
    "B": "textattack/bert-base-uncased-SST-2"
}

client = mlflow.MlflowClient()

for role, model_id in models.items():

    alias = "champion" if role == "A" else "challenger"

    print(f"\nRegistering Model {role}: {model_id}")

    with mlflow.start_run(run_name=f"register_model_{role}"):

        model_info = mlflow.pyfunc.log_model(
            name=f"sentiment_model_{role}",
            python_model=SentimentModel(model_id),
            input_example=pd.DataFrame({
                "text": ["This product is excellent."]
            }),
            pip_requirements=[
                "mlflow==3.18.0",
                "transformers==5.24.0",
                "torch==2.15.0",
                "pandas==3.1.1"
            ]
        )

    registered = mlflow.register_model(
        model_uri=model_info.model_uri,
        name=MODEL_NAME
    )

    client.set_registered_model_alias(
        MODEL_NAME,
        alias,
        str(registered.version)
    )

    print(f"Model {role} registered as version {registered.version}")
    print(f"Alias assigned: {alias}")

print("\nRegistration completed!")
