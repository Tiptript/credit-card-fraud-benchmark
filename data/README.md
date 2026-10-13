# Dataset: Credit Card Fraud Detection

The dataset `creditcard.csv` (~150 MB) is excluded from version control to keep the repository lightweight and adhere to GitHub's best practices.

## 📥 How to Obtain the Dataset

### Option 1: Direct Web Download (Recommended)
1. Download `creditcard.csv` directly from Kaggle:
   👉 **[Kaggle: Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)**
2. Unzip the file and place `creditcard.csv` inside this `data/` folder (or at the repository root).

### Option 2: Using the Kaggle CLI
If you have the Kaggle CLI configured on your machine:
```bash
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ --unzip
```

---

## 📊 Dataset Details
- **Total Transactions:** 284,807
- **Fraudulent Cases:** 492 (0.172% prevalence)
- **Features:** 
  - `Time`: Elapsed seconds since first transaction.
  - `V1` to `V28`: PCA-transformed numerical features (anonymized for client confidentiality).
  - `Amount`: Transaction monetary amount.
  - `Class`: 1 for fraud, 0 for legitimate.
- **Academic Source:**
  > Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, and Gianluca Bontempi.
  > *Calibrating Probability with Undersampling for Unbalanced Classification*.
  > IEEE Symposium Series on Computational Intelligence (SSCI), 2015.
