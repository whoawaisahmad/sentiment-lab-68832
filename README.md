# sentiment-lab-68832
# Legal Notice Classifier

Multi-class text classification of legal notices into 3 categories:
- A: Contract Dispute
- B: Intellectual Property Claim  
- C: Regulatory Compliance

## Setup

pip install -r requirements.txt

## How to Run

1. Open notebooks/sentiment_analysis.ipynb in Kaggle or Jupyter
2. Upload legal_notices.csv to data/raw/
3. Run all cells in order

## Config

All hyperparameters are in config.json:
- random_seed: 42
- test_size: 0.2
- max_features: 5000
- model_1: Logistic Regression (C=1.0)
- model_2: Naive Bayes (alpha=1.0)

## Results

| Model | Vectorizer | Accuracy | Macro F1 |
|-------|-----------|----------|----------|
| LR    | TF-IDF    | best     |          |
| LR    | BoW       |          |          |
| NB    | BoW       |          |          |
| NB    | TF-IDF    | worst    |          |

(Fill in actual numbers from your notebook output)
| Model | Vectorizer | Accuracy | Macro F1 |
|-------|-----------|----------|----------|
| LR    | TF-IDF    | 1.00     | 1.00     |
| LR    | BoW       | 1.00     | 1.00     |
| NB    | BoW       | 1.00     | 1.00     |
| NB    | TF-IDF    | 1.00     | 1.00     |
