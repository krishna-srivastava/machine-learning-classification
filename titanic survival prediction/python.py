import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)

# Load Titanic dataset
df = pd.read_csv("titanic survival prediction/output.csv")
print(df.head())
print(df.isnull().sum())     
print(df.describe())
print(df.info())

df.fillna({
    "age" : df["age"].median(),
    "embarked" : df["embarked"].mode()[0],
},inplace=True)

df.drop("deck",axis=1,inplace=True)
df.drop("class",axis=1,inplace=True)
df.drop("alive",axis=1,inplace=True)
df.drop("embark_town",axis=1,inplace=True)
df.drop("alone",axis=1,inplace=True)
df.drop("adult_male",axis=1,inplace=True)
df.drop("who",axis=1,inplace=True)

print(df.head())
print(df.describe())
print(df.info())

order = {
    'male' : 0,
    'female' : 1
}
df['sex'] = df['sex'].map(order)
df = pd.get_dummies(df, columns=['embarked'], drop_first=True)


df['FamilySize'] = df['sibsp'] + df['parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
df['FareLog'] = np.log1p(df['fare'])

def age_group(age):
    if age < 13:
        return "child"
    elif age < 50:
        return "adult"
    else:
        return "old"

df["Age_group"] = df["age"].apply(age_group)
df["Age_group"] = df["Age_group"].map({"child" : 0,"adult" : 1,"old" : 2})    

def family_type(size):
    if size == 1:
        return 0   # alone
    elif size <= 4:
        return 1   # small family
    else:
        return 2   # large family

df['FamilyType'] = df['FamilySize'].apply(family_type)
df.drop(['sibsp','parch','fare','age'], axis=1, inplace=True)
print(df.head())

# -----------------------------------------------------------------

X = df.drop('survived', axis=1)
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:,1]

baseline = DummyClassifier(strategy='most_frequent')
baseline.fit(X_train_scaled, y_train)
baseline_acc = baseline.score(X_test_scaled, y_test)
print("Baseline Accuracy:", baseline_acc)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix - Titanic Survival Prediction")
plt.show()