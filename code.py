import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")  # adjust encoding if needed

# Drop extra columns if they exist (common in spam datasets)
if len(df.columns) > 2:
    df = df.iloc[:, :2]
df.columns = ["label", "message"]

# 2. Encode labels (ham=0, spam=1)
df["label_num"] = df.label.map({"ham": 0, "spam": 1})

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label_num"], test_size=0.2, random_state=42
)

# 4. Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 5. Train model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 6. Evaluate
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. Save model and vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n✅ Model and vectorizer saved successfully!")
