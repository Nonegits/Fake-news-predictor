# 📰 Fake News Prediction

A Machine Learning project that predicts whether a given news article or news statement is **Fake News** or **Real News** using Natural Language Processing (NLP) and supervised machine learning techniques.

## 📌 Project Overview

The spread of misinformation and fake news has become a major challenge with the rapid growth of digital media and social networking platforms.

This project uses **Natural Language Processing (NLP)** and **Machine Learning** techniques to analyze the textual content of news and classify it as either:

* 🔴 **Fake News**
* 🟢 **Real News**

The trained model can be used to make predictions on new, unseen news content.

---

## 🎯 Objectives

* Detect potentially fake news using machine learning.
* Process and clean textual news data using NLP techniques.
* Convert text into numerical features suitable for ML algorithms.
* Train and evaluate classification models.
* Predict whether new news content is Fake or Real.
* Provide a foundation for deploying the model as an API or web application.

---

## 🛠️ Tech Stack

### Programming Language

* Python

### Libraries & Frameworks

* Pandas
* NumPy
* Scikit-learn
* NLTK
* Matplotlib
* Seaborn
* Joblib

### Machine Learning

* Natural Language Processing (NLP)
* TF-IDF Vectorization
* Text Classification
* Supervised Learning

### Development

* Jupyter Notebook
* VS Code
* Git & GitHub

---

## 📂 Project Structure

```text
Fake-News-Prediction/
│
├── data/
│   ├── raw/
│   │   └── news.csv
│   └── processed/
│
├── notebooks/
│   └── fake_news_prediction.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── predict.py
│
├── models/
│   ├── model.pkl
│   └── vectorizer.pkl
│
├── app/
│   └── app.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

> The project structure may vary depending on the implementation.

---

## 🔄 Machine Learning Workflow

```text
News Dataset
     ↓
Data Collection
     ↓
Data Cleaning
     ↓
Text Preprocessing
     ↓
TF-IDF Feature Extraction
     ↓
Train/Test Split
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Save Trained Model
     ↓
Prediction
```

---

## 🧹 Data Preprocessing

The news text is processed before being provided to the machine learning model.

Typical preprocessing steps include:

1. Handling missing values
2. Removing unnecessary characters
3. Converting text to lowercase
4. Removing punctuation
5. Removing stopwords
6. Tokenization
7. Stemming/Lemmatization

Example:

```text
Original:
"BREAKING!!! Government announces a NEW policy..."

After preprocessing:
"breaking government announces new policy"
```

---

## 🔢 Feature Engineering

Since machine learning algorithms cannot directly understand raw text, the processed news content is converted into numerical features.

This project uses **TF-IDF (Term Frequency–Inverse Document Frequency)** to represent the importance of words within the dataset.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X = vectorizer.fit_transform(text)
```

---

## 🤖 Model Training

The TF-IDF features are provided to a supervised machine learning classifier.

Depending on the implementation, models such as the following can be evaluated:

* Logistic Regression
* Naive Bayes
* Support Vector Machine (SVM)
* Random Forest
* Decision Tree

The model is trained using labeled news data.

```text
Input:
News Article

        ↓

NLP + TF-IDF

        ↓

Machine Learning Model

        ↓

Prediction

        ↓

Fake / Real
```

---

## 📊 Model Evaluation

The trained model can be evaluated using standard classification metrics:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

Example:

```text
Accuracy  : XX%
Precision : XX%
Recall    : XX%
F1-Score  : XX%
```

> Replace the placeholder values with the actual results obtained from your trained model.

---

## 🔮 Prediction

Once the model and TF-IDF vectorizer are trained and saved, they can be used to predict new news content.

Example:

```text
Input:
"Government announces a new economic policy..."

Output:
REAL NEWS
```

Another example:

```text
Input:
"Scientists discover that drinking a particular drink
completely prevents all diseases..."

Output:
FAKE NEWS
```

**Important:** The prediction represents the model's classification based on patterns learned from its training data. It should not be treated as definitive verification of whether a news claim is actually true.

---

## 💾 Saved Model

The trained components can be saved using Joblib:

```python
import joblib

joblib.dump(model, "models/model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
```

They can later be loaded for inference:

```python
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")
```

---

## 🚀 Future Improvements

Possible improvements include:

* Build a FastAPI prediction API.
* Create a simple web interface.
* Add multilingual fake-news detection.
* Experiment with advanced NLP models such as BERT.
* Improve handling of class imbalance.
* Add model monitoring.
* Track experiments using MLflow.
* Add automated data and model pipelines.
* Deploy the model using Docker and cloud services.

---

## ⚠️ Limitations

Fake-news detection is a challenging NLP problem.

A machine learning classifier generally learns patterns from the dataset rather than independently verifying facts against authoritative sources. Therefore:

* Predictions can be incorrect.
* Dataset quality strongly affects performance.
* Biased or outdated training data can affect predictions.
* A model prediction should not be considered a fact-check by itself.

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Fake-News-Prediction.git
```

Navigate to the project:

```bash
cd Fake-News-Prediction
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

If using the Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
notebooks/fake_news_prediction.ipynb
```

Run the cells sequentially to preprocess the data, train the model, evaluate its performance, and generate predictions.

---

## 👨‍💻 Author

**Your Name**

B.Tech — Computer Science & Engineering (AI & ML)

---

## 📜 Disclaimer

This project is developed for **educational and research purposes**. The predictions generated by the model should not be considered authoritative fact-checking or used as the sole basis for decisions regarding the truthfulness of real-world news.

---
