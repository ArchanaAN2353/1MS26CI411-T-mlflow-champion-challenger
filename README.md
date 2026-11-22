\# MLflow Champion–Challenger Model Evaluation



\*\*Student USN:\*\* 1MS26CI411-T

\*\*Student Name:\*\* Archana A N  

\*\*Project:\*\* Sentiment Analysis using MLflow  

\*\*Task:\*\* Text Classification – Sentiment Analysis



\---



\## 1. Project Overview



This project implements a Champion–Challenger model evaluation system using MLflow and Hugging Face Transformers.



Two pre-trained sentiment analysis models are evaluated using the same dataset. Their performance is compared using classification metrics and inference latency.



The better-performing model is selected as the Champion, while the other model is assigned the Challenger role.



MLflow is used for experiment tracking, metric logging, model registration, model versioning, alias management and prediction.



\## 2. Objectives



\- Evaluate two pre-trained sentiment analysis models.

\- Compare model performance using the same evaluation dataset.

\- Track experiments and metrics using MLflow.

\- Register both models in the MLflow Model Registry.

\- Assign Champion and Challenger aliases.

\- Load the Champion model directly from MLflow.

\- Generate predictions on new, unseen input reviews.

\- Visualize model performance and inference latency.



\## 3. Models Used



Two Hugging Face pre-trained models were selected.



| Role | Model |

|---|---|

| Model A | distilbert-base-uncased-finetuned-sst-2-english |

| Model B | textattack/bert-base-uncased-SST-2 |



Both models perform binary sentiment classification:



\- Positive sentiment

\- Negative sentiment



\## 4. Dataset



A custom sentiment analysis dataset containing 20 labelled reviews was created.



\- 10 Positive reviews

\- 10 Negative reviews



The dataset is stored in:



`data/eval\_inputs.csv`



A separate dataset containing five new reviews is used to test the registered Champion model.



`data/new\_data.csv`



\## 5. Evaluation Metrics



The following metrics were used to compare the two models:



\- \*\*Accuracy:\*\* Measures the overall proportion of correct predictions.

\- \*\*Precision:\*\* Measures how many predicted positive reviews were actually positive.

\- \*\*Recall:\*\* Measures how many actual positive reviews were correctly identified.

\- \*\*F1-score:\*\* Combines precision and recall.

\- \*\*Average Latency:\*\* Measures average inference time per review in milliseconds.

\- \*\*Total Inference Time:\*\* Measures the total time taken to process the evaluation dataset.



\## 6. Champion Selection Strategy



The decision rule was defined before final model selection.



1\. The model with the higher F1-score is selected.

2\. If the F1-score difference is within 0.01, the model with lower average inference latency is selected.



This strategy considers both prediction quality and inference speed.



\## 7. Experimental Results



Both models achieved the same classification performance on the evaluation dataset.



| Metric | Model A | Model B |

|---|---:|---:|

| Accuracy | 1.00 | 1.00 |

| Precision | 1.00 | 1.00 |

| Recall | 1.00 | 1.00 |

| F1-score | 1.00 | 1.00 |

| Average Latency | 13.63 ms | 21.06 ms |

| Total Inference Time | 0.273 s | 0.421 s |



\### Final Decision



\*\*Champion:\*\* Model A – DistilBERT



\*\*Challenger:\*\* Model B – BERT



Both models achieved an F1-score of 1.00. Since their F1-score difference was within 0.01, Model A was selected as the Champion because it had lower average inference latency.



\*Note: These results are based on the small custom evaluation dataset and should not be interpreted as a guarantee of performance on larger real-world datasets.\*



\## 8. MLflow Experiment Tracking



MLflow was used to track the experiments.



Experiment name:



`Sentiment Analysis Champion Challenger`



The following information was logged:



\- Model identifiers and evaluation parameters

\- Accuracy, precision, recall and F1-score

\- Average inference latency

\- Total inference time

\- Student USN tag

\- Prediction CSV files

\- Model comparison graphs

\- Final Champion decision



\## 9. Model Registry



Both models were registered under the same registered model name:



`SentimentAnalysisClassifier`



| Model | Version | Alias |

|---|---:|---|

| Model A | 1 | champion |

| Model B | 2 | challenger |



The Champion model is loaded using its MLflow alias:



`models:/SentimentAnalysisClassifier@champion`



This allows the prediction script to load the selected model through MLflow rather than directly loading the Hugging Face model.



\## 10. Champion Model Predictions



The registered Champion model was loaded from MLflow and tested using five new reviews.



The output contains:



\- Input review text

\- Predicted sentiment

\- Prediction confidence score



The predictions are saved in:



`outputs/champion\_predictions.csv`



\## 11. Visualizations



Two graphs were generated to compare the models.



\*\*Figure 1: Performance Comparison\*\*



`outputs/performance\_comparison.png`



Compares accuracy, precision, recall and F1-score.



\*\*Figure 2: Inference Latency Comparison\*\*



`outputs/latency\_comparison.png`



Compares average inference latency of both models.



Both graphs were also logged as MLflow artifacts.



\## 12. Project Structure



```text

MLflow\_Champion\_Challenger/

│

├── data/

│   ├── eval\_inputs.csv

│   └── new\_data.csv

│

├── outputs/

│   ├── predictions\_model\_A.csv

│   ├── predictions\_model\_B.csv

│   ├── metrics\_summary.csv

│   ├── decision.json

│   ├── champion\_predictions.csv

│   ├── performance\_comparison.png

│   └── latency\_comparison.png

│

├── evaluate.py

├── register\_champion.py

├── predict\_from\_mlflow.py

├── generate\_plots.py

├── log\_artifacts.py

├── requirements.txt

└── README.md

```



\## 13. Technologies Used



\- Python

\- MLflow

\- Hugging Face Transformers

\- PyTorch

\- Pandas

\- Scikit-learn

\- Matplotlib



\## 14. How to Run the Project



\### Step 1: Install dependencies



```bash

pip install -r requirements.txt

```



\### Step 2: Start the MLflow tracking server



```bash

mlflow server --host 127.0.0.1 --port 5000

```



\### Step 3: Evaluate both models



```bash

python evaluate.py

```



\### Step 4: Register both models



```bash

python register\_champion.py

```



\### Step 5: Generate Champion predictions



```bash

python predict\_from\_mlflow.py

```



\### Step 6: Generate comparison graphs



```bash

python generate\_plots.py

```



\### Step 7: Log artifacts into MLflow



```bash

python log\_artifacts.py

```



Open the MLflow dashboard:



http://127.0.0.1:5000



\## 15. Conclusion



This project demonstrates how MLflow can be used to implement a Champion–Challenger workflow for machine learning model evaluation.



Two pre-trained sentiment analysis models were evaluated using identical input data. Based on the predefined decision rule, DistilBERT was selected as the Champion due to its lower inference latency while maintaining the same F1-score.



Both models were registered in MLflow with their respective aliases. The Champion model was successfully loaded from the Model Registry and used to generate predictions on new reviews.



The project provides practical experience with experiment tracking, model evaluation, model registration, versioning and model deployment workflows.

