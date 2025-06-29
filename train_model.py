import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import pickle

df = pd.read_csv('disease_dataset.csv')
X = df.drop('prognosis', axis=1)
y = LabelEncoder().fit_transform(df['prognosis'])

model = MultinomialNB()
model.fit(X, y)

pickle.dump((model, X.columns.tolist(), y), open('model.pkl', 'wb'))
print("✅ Model trained and saved to model.pkl")
