import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

# 1. Pré-processamento e Engenharia de Atributos

def advanced_preprocessing(df):
    df = df.copy()
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', \
                                     'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    df['Cabin'] = df['Cabin'].fillna('U')
    df['Deck'] = df['Cabin'].map(lambda x: x[0])
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
    
    df['Age'] = df.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    
    return df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# PROTEÇÃO PARA WINDOWS 
if __name__ == "__main__":
    try:
        train = pd.read_csv('train.csv')
        test = pd.read_csv('test.csv')
        
        y = train['Survived']
        X = advanced_preprocessing(train).drop(['Survived', 'PassengerId'], axis=1)
        test_id = test['PassengerId']
        X_test = advanced_preprocessing(test).drop(['PassengerId'], axis=1)

        categorical_cols = ['Sex', 'Embarked', 'Title', 'Deck']
        for col in categorical_cols:
            le = LabelEncoder()
            combined = pd.concat([X[col], X_test[col]], axis=0)
            le.fit(combined)
            X[col] = le.transform(X[col])
            X_test[col] = le.transform(X_test[col])

        param_grid = {
            'n_estimators': [100, 300],
            'max_depth': [5, 8],
            'min_samples_leaf': [2, 4]
        }

        print("Treinando o modelo... Aguarde alguns segundos.")
        # Mudei n_jobs para None para evitar o erro de travamento
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=None)
        grid_search.fit(X, y)

        print(f"Melhor Acurácia: {grid_search.best_score_:.4f}")

        final_preds = grid_search.predict(X_test)
        submission = pd.DataFrame({'PassengerId': test_id, 'Survived': final_preds})
        submission.to_csv('submission.csv', index=False)
        print("Arquivo 'submission.csv' criado com sucesso!")

    except FileNotFoundError:
        print("Erro: Verifique se os arquivos CSV estão na mesma pasta.")

        # Optei pelo Random Forest por ser um modelo de ensemble robusto que lida bem com variáveis categóricas e oferece um excelente equilíbrio entre complexidade e poder preditivo para este problema clássico."