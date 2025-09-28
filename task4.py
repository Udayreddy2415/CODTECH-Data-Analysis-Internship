# ===========================================
# Internship Task - 4: Sentiment Analysis
# Using NLP + Machine Learning
# Deliverable: Notebook-style script
# ===========================================

# Install required libraries (uncomment if needed)
# !pip install scikit-learn nltk matplotlib seaborn

import re
import nltk
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

# Download NLTK stopwords (first run only)
nltk.download('stopwords')
from nltk.corpus import stopwords

# -------------------------------
# Step 1: Sample Textual Dataset
# -------------------------------
# Replace this with your own dataset (tweets, reviews, etc.)
data = {
    "review": [
        "I loved the movie, it was fantastic and inspiring!",
        "Absolutely terrible film. Waste of time.",
        "The plot was engaging and the actors were brilliant.",
        "Worst movie I have ever seen. Horrible acting.",
        "It was okay, not the best but watchable.",
        "A masterpiece. Truly moving and beautiful.",
        "I hated this movie, very boring and predictable.",
        "An excellent film with stunning visuals!",
        "Mediocre at best, wouldn’t recommend.",
        "Loved it! Great direction and story."
    ],
    "sentiment": [1, 0, 1, 0, 1, 1, 0, 1, 0, 1]  # 1=Positive, 0=Negative
}
df = pd.DataFrame(data)

# -------------------------------
# Step 2: Preprocessing Function
# -------------------------------
stop_words = set(stopwords.words("english"))

def preprocess(text):
    # Lowercase
    text = text.lower()
    # Remove special characters/numbers
    text = re.sub(r"[^a-zA-Z\\s]", "", text)
    # Remove stopwords
    tokens = [word for word in text.split() if word not in stop_words]
    return " ".join(tokens)

df["clean_review"] = df["review"].apply(preprocess)

# -------------------------------
# Step 3: Feature Extraction (TF-IDF)
# -------------------------------
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df["clean_review"])
y = df["sentiment"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# Step 4: Train ML Model
# -------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -------------------------------
# Step 5: Evaluation
# -------------------------------
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative","Positive"], yticklabels=["Negative","Positive"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -------------------------------
# Step 6: Insights
# Most influential words for sentiment
# -------------------------------
feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]

top_positive_idx = coefficients.argsort()[-10:][::-1]
top_negative_idx = coefficients.argsort()[:10]

print("\\nTop Positive Words:")
for i in top_positive_idx:
    print(f"{feature_names[i]} ({coefficients[i]:.3f})")

print("\\nTop Negative Words:")
for i in top_negative_idx:
    print(f"{feature_names[i]} ({coefficients[i]:.3f})")
