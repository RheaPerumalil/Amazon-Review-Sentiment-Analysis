# Sentiment Analysis of Amazon Product Reviews

This project analyzes 1.5 million Amazon product reviews across Grocery, Clothing, and Electronics categories to classify sentiment as **positive**, **neutral**, or **negative**.  
We implemented and compared multiple machine learning models — including VADER, Logistic Regression, SVM, Random Forest, and XGBoost — with special focus on handling class imbalance and improving neutral sentiment detection.

**View the full notebook**: [Open in Google Colab](https://colab.research.google.com/drive/1zBbGkSWcJreTrfmgQ0ybk0fbwXmIEPhf?usp=sharing)

---

## Project Highlights

- Preprocessed over 600,000 cleaned reviews using NLP techniques (NLTK, spaCy)
- Engineered features using TF-IDF with n-grams and frequency filtering
- Handled class imbalance using class weighting, SMOTE, and stratified downsampling
- Built and tuned ML models with GridSearchCV and custom weighting
- Evaluated models using macro-averaged Precision, Recall, and F1 Score

---

## Dataset

- **Source**: [Amazon Reviews 2023 Dataset](https://amazon-reviews-2023.github.io/)
- **Size**: 1.5 million reviews, cleaned to ~640,000
- **Categories**: Grocery, Clothing, Electronics
- **Sentiment Labels**:
  - 1–2 stars → Negative
  - 3 stars → Neutral
  - 4–5 stars → Positive

---

## Tools and Libraries

- Python (Colab)
- Pandas, NumPy
- NLTK, spaCy
- Scikit-learn
- XGBoost
- Matplotlib, Seaborn

---

## Models Compared

| Model                | Accuracy | Macro F1 |
|---------------------|----------|----------|
| VADER (baseline)    | 0.81     | 0.49     |
| Logistic Regression | 0.81     | 0.62     |
| SVM (weighted)      | 0.87     | 0.64     |
| Random Forest       | 0.86     | 0.54     |
| XGBoost (tuned)     | 0.87     | 0.59     |

SVM with manually tuned class weights performed best across all metrics, though neutral sentiment remained the most difficult to classify.

---

## Key Challenges and Insights

- Neutral sentiment is hard to detect due to vague language and class overlap
- Many models favored the dominant positive class
- Class reweighting and careful preprocessing significantly improved balance

---

## Future Work

- Integrate BERT or other transformer models for context-aware sentiment detection
- Explore ensemble methods to boost minority class performance
- Build a Streamlit dashboard for interactive sentiment monitoring

---

## Team

- Cristian Aponte  
- Sandip Singh  
- Rhea Perumalil  

---

