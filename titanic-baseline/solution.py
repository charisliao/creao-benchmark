#!/usr/bin/env python3
"""
solution.py

Titanic ML pipeline:
1. Load data
2. Clean & impute
3. Feature engineering
4. Encode
5. Train & evaluate
6. Predict & save submission
"""

import pandas as pd                                            # data handling :contentReference[oaicite:5]{index=5}
import numpy as np                                             # numerical ops
from sklearn.linear_model import LogisticRegression            # baseline classifier :contentReference[oaicite:6]{index=6}
from sklearn.ensemble import RandomForestClassifier            # ensemble method :contentReference[oaicite:7]{index=7}
from sklearn.model_selection import cross_val_score

# 1. Load Data
train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')

# 2. Missing Data Overview
print(train.isnull().sum()) 

# 3. Imputation Helpers
def impute_age(df):
    df['Title'] = df.Name.str.extract(r',\s*([^\.]+)\.')
    rare_titles = ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona']
    df['Title'].replace(rare_titles, 'Rare', inplace=True)
    df['Title'].replace({'Mlle':'Miss','Ms':'Miss','Mme':'Mrs'}, inplace=True)
    df['Age'] = df.groupby(['Sex','Pclass'])['Age']\
                  .apply(lambda grp: grp.fillna(grp.median()))
    return df

def impute_embarked(df):
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    return df

def impute_fare(df):
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    return df

for df in (train, test):
    df = impute_age(df)
    df = impute_embarked(df)
    df = impute_fare(df)

# 4. Feature Engineering
for df in (train, test):
    df['FamilySize'] = df.SibSp + df.Parch + 1
    df['IsAlone']     = (df.FamilySize == 1).astype(int)
    df['AgeBin']      = pd.cut(df.Age, 5, labels=False)
    df['FareBin']     = pd.qcut(df.Fare, 4, labels=False)

# 5. Encoding
mapping = {
    'Sex':      {'male':0, 'female':1},
    'Embarked': {'S':0, 'C':1, 'Q':2},
    'Title':    {'Mr':1,'Miss':2,'Mrs':3,'Master':4,'Rare':5}
}
for df in (train, test):
    df.replace(mapping, inplace=True)

# Select features
features = ['Pclass','Sex','AgeBin','FareBin','Embarked','FamilySize','IsAlone','Title']

X_train = train[features]
y_train = train.Survived
X_test  = test[features]

# 6. Modeling & Evaluation
lr = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

lr_score = cross_val_score(lr, X_train, y_train, cv=5).mean()
rf_score = cross_val_score(rf, X_train, y_train, cv=5).mean()

print(f'Logistic Regression CV Accuracy: {lr_score:.4f}')
print(f'Random Forest CV Accuracy:       {rf_score:.4f}')

# 7. Fit & Predict with best model (Random Forest here)
rf.fit(X_train, y_train)
preds = rf.predict(X_test)

submission = pd.DataFrame({
    'PassengerId': test.PassengerId,
    'Survived':    preds.astype(int)
})
submission.to_csv('submission.csv', index=False)
print('Saved submission.csv')
