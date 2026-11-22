# MLflow Champion–Challenger Model Evaluation

### Sentiment Analysis using Hugging Face Transformers and MLflow

![Python](https://img.shields.io/badge/Python-3.14-blue)
![MLflow](https://img.shields.io/badge/MLflow-3.18-orange)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow)
![Project](https://img.shields.io/badge/Project-Champion--Challenger-green)

---

## Student Information

| Field | Details |
|---|---|
| **Student Name** | Archana A N |
| **Student USN** | 1MS26CI411-T |
| **Section** | A |
| **College** | M S Ramaiah Institute of Technology |
| **Project** | Sentiment Analysis using MLflow |
| **Task** | Text Classification – Sentiment Analysis |

---

## 1. Project Overview

This project implements a **Champion–Challenger model evaluation workflow** using MLflow and Hugging Face Transformers.

Two pre-trained sentiment analysis models are evaluated using the same labelled dataset. Their performance is compared using classification metrics and inference latency.

The better-performing model is selected as the Champion, while the other model is assigned the Challenger role.

MLflow is used for experiment tracking, metric logging, model registration, model versioning, alias management and prediction.

The selected Champion model is subsequently loaded through its MLflow alias and used to predict the sentiment of new reviews.

## 2. Project Objectives

- Evaluate two pre-trained sentiment analysis models.
- Compare model performance using identical evaluation data.
- Measure accuracy, precision, recall and F1-score.
- Compare average inference latency and total inference time.
- Track experiments and metrics using MLflow.
- Register both models under a common registered model name.
- Assign Champion and Challenger aliases.
- Load the Champion model using its MLflow alias.
- Generate predictions on new review inputs.
- Visualize and log model comparison results.

## 3. Models Used

Two pre-trained Hugging Face models were selected.

| Model | Hugging Face Model ID |
|---|---|
| **Model A** | `distilbert-base-uncased-finetuned-sst-2-english` |
| **Model B** | `textattack/bert-base-uncased-SST-2` |

Both models perform binary sentiment classification:

- **Positive (1)**
- **Negative (0)**

Both models were evaluated on the same input samples to ensure a fair comparison.

## 4. Dataset Description

A custom evaluation dataset containing **20 labelled reviews** was created.

| Sentiment | Number of Reviews |
|---|---:|
| Positive | 10 |
| Negative | 10 |
| **Total** | **20** |

The evaluation dataset is stored in:

`data/eval_inputs.csv`

A separate dataset containing five new reviews is used to test the registered Champion model:

`data/new_data.csv`

The new reviews are used for prediction after model registration.

## 5. Evaluation Metrics

The following metrics were used to compare both models.

| Metric | Description |
|---|---|
| Accuracy | Proportion of correctly classified reviews |
| Precision | Proportion of predicted positive reviews that are actually positive |
| Recall | Proportion of actual positive reviews correctly identified |
| F1-score | Harmonic mean of precision and recall |
| Average Latency | Average inference time per review in milliseconds |
| Total Inference Time | Total time taken to process the evaluation dataset |

These metrics help evaluate both classification performance and inference efficiency.

## 6. Champion Selection Strategy

A predefined decision rule was used to select the Champion model.

1. Select the model with the higher F1-score.
2. If the F1-score difference is within 0.01, select the model with lower average inference latency.

This approach considers both prediction quality and inference speed.

## 7. Experimental Results

Both models achieved identical classification metrics on the evaluation dataset.

| Metric | Model A | Model B |
|---|---:|---:|
| Accuracy | 1.00 | 1.00 |
| Precision | 1.00 | 1.00 |
| Recall | 1.00 | 1.00 |
| F1-score | 1.00 | 1.00 |
| Average Latency | 13.63 ms | 21.06 ms |
| Total Inference Time | 0.273 s | 0.421 s |

### Final Model Selection

| Role | Selected Model |
|---|---|
| **Champion** | Model A – DistilBERT |
| **Challenger** | Model B – BERT |

**Reason:** Both models achieved the same F1-score. Since the difference was within 0.01, Model A was selected because it had lower average inference latency.

### 7.1 Model Comparison and Trade-off Analysis

Both models achieved identical classification results on the 20 evaluation samples, with 100% accuracy, precision, recall and F1-score. Therefore, the Challenger model did not demonstrate an improvement in prediction quality over the Champion model.

However, a difference was observed in inference speed. Model A recorded an average latency of 13.63 ms, whereas Model B recorded 21.06 ms. This means Model B was approximately 54.5% slower than Model A on the same evaluation dataset. Since both models achieved the same F1-score, Model A was selected as the Champion according to the predefined decision rule.

The results also showed that both models predicted the same sentiment labels for all 20 samples. Although their confidence scores differed, there were no prediction disagreements. Consequently, five disagreement examples could not be identified from this dataset.

The evaluation dataset contains only 20 reviews, with 10 positive and 10 negative samples. This small sample size limits the reliability and generalizability of the results. A larger and more diverse dataset would be needed for a stronger comparison.

Overall, Model A provides the better latency-performance trade-off in this experiment while maintaining the same classification metrics as Model B.

### 7.2 Qualitative Comparison

Both models produced identical predicted sentiment labels for all 20 evaluation examples.

- Total examples compared: **20**
- Prediction disagreements: **0**
- Both models achieved: **100% accuracy**

No disagreement examples were found. Therefore, five disagreement cases could not be provided without inventing results.

The confidence scores differed between models even though their final predicted labels were identical.

The detailed qualitative comparison is available in:

`outputs/qualitative_analysis.txt`

**Note:** The evaluation results are based on a small custom dataset and do not guarantee equivalent performance on larger real-world datasets.

## 8. MLflow Experiment Tracking

MLflow was used to track the model evaluation experiments.

**Experiment Name:**

`Sentiment Analysis Champion Challenger`

The following information was recorded:

- Model identifiers and evaluation parameters.
- Accuracy, precision, recall and F1-score.
- Average inference latency.
- Total inference time.
- Student USN tag.
- Model prediction files.
- Comparison graphs.
- Final Champion selection decision.
- Evaluation dataset and summary artifacts.

The final project artifacts were logged under the run:

`final_project_artifacts`

## 9. MLflow Model Registry

Both models were registered under the same registered model name:

`SentimentAnalysisClassifier`

| Model | Version | Alias |
|---|---:|---|
| Model A – DistilBERT | 1 | `champion` |
| Model B – BERT | 2 | `challenger` |

The Champion model is loaded using the following MLflow model URI:

```python
models:/SentimentAnalysisClassifier@champion
```

Using aliases allows the selected model to be accessed through MLflow without hardcoding its version number in the prediction workflow.

## 10. Champion Model Predictions

After model registration, the Champion model was loaded directly from the MLflow Model Registry.

Five new review inputs were used to generate predictions.

| Input Review | Predicted Sentiment |
|---|---|
| Amazing product | Positive |
| Poor quality | Negative |
| Excellent service | Positive |
| Unhappy purchase | Negative |
| Works perfectly | Positive |

The prediction output includes:

- Input review text.
- Predicted sentiment.
- Prediction confidence score.

The predictions are saved in:

`outputs/champion_predictions.csv`

## 11. Visualizations

Two graphs were generated to compare the performance of both models.

### Figure 1: Performance Comparison

File: `outputs/performance_comparison.png`

This graph compares:

- Accuracy
- Precision
- Recall
- F1-score

Both models achieved identical classification scores.

### Figure 2: Inference Latency Comparison

File: `outputs/latency_comparison.png`

This graph compares the average inference latency of both models.

Model A demonstrated lower average inference latency than Model B.

Both graphs were also logged as MLflow artifacts.

## 12. MLflow Artifacts

The following project artifacts were logged in MLflow:

| Artifact | Description |
|---|---|
| `performance_comparison.png` | Model performance graph |
| `latency_comparison.png` | Inference latency graph |
| `metrics_summary.csv` | Evaluation metrics in CSV format |
| `metrics_summary.json` | Evaluation metrics in JSON format |
| `decision.json` | Final Champion selection |
| `champion_predictions.csv` | Predictions from the Champion model |
| `qualitative_analysis.txt` | Qualitative comparison of both models |
| `eval_inputs.csv` | Evaluation dataset |

## 13. Project Structure

```text
MLflow_Champion_Challenger/
│
├── data/
│   ├── eval_inputs.csv
│   └── new_data.csv
│
├── outputs/
│   ├── predictions_model_A.csv
│   ├── predictions_model_B.csv
│   ├── metrics_summary.csv
│   ├── metrics_summary.json
│   ├── decision.json
│   ├── champion_predictions.csv
│   ├── qualitative_analysis.txt
│   ├── performance_comparison.png
│   └── latency_comparison.png
│
├── screenshots/
│
├── evaluate.py
├── register_champion.py
├── predict_from_mlflow.py
├── generate_plots.py
├── log_artifacts.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

## 14. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| MLflow | Experiment tracking and model registry |
| Hugging Face Transformers | Pre-trained sentiment analysis models |
| PyTorch | Model execution |
| Pandas | Dataset processing |
| Scikit-learn | Evaluation metrics |
| Matplotlib | Data visualization |
| Git and GitHub | Version control and project hosting |

## 15. How to Run the Project

### Step 1: Clone the repository

```bash
git clone https://github.com/ArchanaAN2353/1MS26CI411-T-mlflow-champion-challenger.git
```

### Step 2: Navigate to the project directory

```bash
cd 1MS26CI411-T-mlflow-champion-challenger
```

### Step 3: Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Start the MLflow tracking server

```bash
python -m mlflow server --host 127.0.0.1 --port 5000
```

Open the MLflow dashboard:

`http://127.0.0.1:5000`

### Step 6: Evaluate both models

```bash
python evaluate.py
```

### Step 7: Register both models

```bash
python register_champion.py
```

### Step 8: Generate Champion predictions

```bash
python predict_from_mlflow.py
```

### Step 9: Generate comparison graphs

```bash
python generate_plots.py
```

### Step 10: Log artifacts into MLflow

```bash
python log_artifacts.py
```

**Note:** The evaluation and registration steps require access to the Hugging Face model downloads. The MLflow tracking server should remain running while executing the MLflow-related scripts.

## 16. Conclusion

This project demonstrates the implementation of a Champion–Challenger workflow for sentiment analysis using MLflow and Hugging Face Transformers.

Two pre-trained models were evaluated using the same dataset and compared using classification metrics and inference latency.

Both models achieved identical classification performance on the evaluation dataset. Based on the predefined decision rule, DistilBERT was selected as the Champion because it demonstrated lower average inference latency.

Both models were successfully registered in MLflow with their respective aliases. The Champion model was then loaded from the Model Registry and used to generate predictions on five new reviews.

The project provided practical experience in model evaluation, experiment tracking, artifact logging, model registration, model versioning, alias management and prediction workflows.

---

## GitHub Repository

**Project Repository:**

https://github.com/ArchanaAN2353/1MS26CI411-T-mlflow-champion-challenger

