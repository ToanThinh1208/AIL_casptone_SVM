# Sentiment Analysis on Amazon Product Reviews

This project implements and compares various Machine Learning models for sentiment analysis on Amazon Product Reviews. The project covers both English and Vietnamese datasets, utilizing advanced text embedding techniques and standard classification algorithms.

## Project Overview

The goal of this project is to classify product reviews into positive or negative sentiments. It involves a complete pipeline from data cleaning and preprocessing to model training and evaluation.

**Key Features:**
*   **Multi-language Support:** Analysis performed on both English and Vietnamese datasets.
*   **Advanced Embeddings:** Utilizes **Sentence Transformers** (specifically `all-mpnet-base-v2`) for high-quality text representation, moving beyond traditional TF-IDF or Word2Vec approaches.
*   **Model Comparison:** Evaluates multiple models including **Logistic Regression**, **Support Vector Machines (SVM)**, and **XGBoost**.
*   **Data Balancing:** Implements techniques to handle class imbalance (downsampling).

## Dataset Description

The project uses the **Amazon Product Review** dataset.

*   **Source:**
    *   English: `test_English/dataset_en/Amazon_Product_Review_full_en.csv`
    *   Vietnamese: `test_vietnamese/dataset_vi/Amazon_Product_Review_full_vi.csv`
*   **Structure:** The raw data typically contains star ratings (1-5), review headlines, and review bodies.
*   **Target Variable:** A binary `sentiment` label (0 for negative, 1 for positive).
*   **Preprocessing:**
    *   **Cleaning:** Removal of HTML tags, special characters, and numbers. Text is converted to lowercase.
    *   **Tokenization & Lemmatization:** Applied using `nltk` for English and `underthesea` for Vietnamese.
    *   **Combination:** `review_headline` and `review_body` are combined to form a `full_review` text for better context.
    *   **Balancing:** The dataset is balanced to ensure an equal number of positive and negative samples (e.g., ~5000 samples per class in the English subset used).

## Methodology

### 1. Data Cleaning & Preprocessing
*   **English:** Uses `nltk` for stopword removal and lemmatization.
*   **Vietnamese:** Uses `underthesea` for Vietnamese-specific text processing.
*   **Cleaning:** Regex-based cleaning to remove noise.

### 2. Feature Extraction
*   **Sentence Transformers:** The project leverages the `sentence-transformers` library.
*   **Model:** `all-mpnet-base-v2` is used to generate dense vector representations (embeddings) of the reviews. This model is state-of-the-art for semantic similarity and clustering tasks, providing rich contextual features for the classifiers.

### 3. Machine Learning Models
The following models are trained and tuned using `GridSearchCV` for optimal hyperparameters:

*   **Logistic Regression:** A robust baseline model.
*   **Support Vector Machine (SVM):** Effective for high-dimensional spaces (like text embeddings). Uses RBF kernel.
*   **XGBoost:** A powerful gradient boosting algorithm known for high performance.

## Results & Performance

The models were evaluated on a held-out test set (20% split). Below is a summary of the accuracy achieved:

### English Dataset
| Model | Accuracy | Best Parameters |
| :--- | :--- | :--- |
| **Logistic Regression** | **~87.07%** | `C=10`, `penalty='l2'`, `solver='lbfgs'` |
| **SVM** | **~87.02%** | `C=1`, `gamma='scale'`, `kernel='rbf'` |
| **XGBoost** | ~85.59% | `n_estimators=100`, `max_depth=6`, `learning_rate=0.1` |

### Vietnamese Dataset
| Model | Accuracy | Best Parameters |
| :--- | :--- | :--- |
| **SVM** | **~85.86%** | `C=1`, `gamma='scale'`, `kernel='rbf'` |

*Note: Results indicate that Logistic Regression and SVM perform comparably well, slightly outperforming XGBoost on this specific embedding set.*

## Installation & Usage

### Prerequisites
Ensure you have Python 3.10+ installed. Install the required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn wordcloud
pip install sentence-transformers xgboost imbalanced-learn
pip install nltk underthesea
```

### Project Structure
```
AIL_casptone_SVM/
├── ML/
│   ├── data_cleaning.ipynb       # Data cleaning for English dataset
│   ├── data_analysis.ipynb       # EDA for English dataset
│   ├── logistic.ipynb            # Logistic Regression model (English)
│   ├── model_svm.ipynb           # SVM model (English)
│   ├── model_xgboost.ipynb       # XGBoost model (English)
│   ├── test_English/             # English dataset directory
│   └── test_vietnamese/          # Vietnamese dataset & models
│       ├── data_cleaning_vi1.ipynb
│       ├── data_analysis_vi1.ipynb
│       └── model_svm_v1i.ipynb   # SVM model (Vietnamese)
```

### Running the Project
1.  **Data Cleaning:** Run `data_cleaning.ipynb` (or `data_cleaning_vi1.ipynb`) to generate the cleaned CSV files.
2.  **Analysis:** Run `data_analysis.ipynb` to visualize data distributions and word clouds.
3.  **Modeling:** Run any of the model notebooks (e.g., `model_svm.ipynb`) to train the model and see the evaluation results.

## Future Improvements
*   **Deep Learning:** Implement BERT or LSTM models as suggested in the project notes for potentially better performance.
*   **Multilingual Embeddings:** Experiment with `paraphrase-multilingual-mpnet-base-v2` specifically for the Vietnamese dataset to improve embedding quality.
*   **Hyperparameter Tuning:** Expand the grid search for XGBoost to potentially improve its performance.
