import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def process_data_and_train_model(train_path, test_path):
    # Load data
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # Store test passenger IDs for submission
    test_ids = test['PassengerId']

    # --- Data Cleaning & Feature Engineering (from your notebook) ---
    # Drop unnecessary columns
    train = train.drop(['Cabin', 'Ticket'], axis=1)
    test = test.drop(['Cabin', 'Ticket'], axis=1)

    # Fill missing 'Embarked'
    train = train.fillna({"Embarked": "S"})

    # Fill missing 'Age' with a placeholder and create AgeGroup
    train["Age"] = train["Age"].fillna(-0.5)
    test["Age"] = test["Age"].fillna(-0.5)
    bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
    labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    train['AgeGroup'] = pd.cut(train["Age"], bins, labels=labels)
    test['AgeGroup'] = pd.cut(test["Age"], bins, labels=labels)

    # Create 'Title' feature from Name
    combine = [train, test]
    for dataset in combine:
        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\\.', expand=False)
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
        dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    # Map titles to numbers
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}
    for dataset in combine:
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)

    # Fill 'Unknown' AgeGroup based on Title
    age_title_mapping = {1: "Young Adult", 2: "Student", 3: "Adult", 4: "Baby", 5: "Adult", 6: "Adult"}
    for x in range(len(train["AgeGroup"])):
        if train["AgeGroup"][x] == "Unknown":
            train["AgeGroup"][x] = age_title_mapping[train["Title"][x]]
    for x in range(len(test["AgeGroup"])):
        if test["AgeGroup"][x] == "Unknown":
            test["AgeGroup"][x] = age_title_mapping[test["Title"][x]]
    
    # Map AgeGroup to numbers
    age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}
    train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
    test['AgeGroup'] = test['AgeGroup'].map(age_mapping)
    
    # Drop original 'Age' and 'Name'
    train = train.drop(['Age', 'Name'], axis=1)
    test = test.drop(['Age', 'Name'], axis=1)

    # Map 'Sex' and 'Embarked' to numbers
    sex_mapping = {"male": 0, "female": 1}
    train['Sex'] = train['Sex'].map(sex_mapping)
    test['Sex'] = test['Sex'].map(sex_mapping)
    
    embarked_mapping = {"S": 1, "C": 2, "Q": 3}
    train['Embarked'] = train['Embarked'].map(embarked_mapping)
    test['Embarked'] = test['Embarked'].map(embarked_mapping)

    # Fill missing 'Fare' in test set
    test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)

    # Create 'FareBand'
    train['FareBand'] = pd.qcut(train['Fare'], 4, labels=[1, 2, 3, 4])
    test['FareBand'] = pd.qcut(test['Fare'], 4, labels=[1, 2, 3, 4])

    # Drop original 'Fare'
    train = train.drop(['Fare'], axis=1)
    test = test.drop(['Fare'], axis=1)

    # --- Model Training ---
    predictors = train.drop(['Survived', 'PassengerId'], axis=1)
    target = train["Survived"]
    
    randomforest = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    randomforest.fit(predictors, target)

    return train, test, randomforest, test_ids