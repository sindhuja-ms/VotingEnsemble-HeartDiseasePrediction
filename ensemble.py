import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def train_voting_ensemble():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    df = pd.read_csv(url, names=columns, na_values='?')
    df.fillna(df.mean(), inplace=True)

    X = df.drop('target', axis=1)
    y = (df['target'] > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf1 = LogisticRegression(max_iter=1000, random_state=42)
    clf2 = KNeighborsClassifier()
    clf3 = SVC(probability=True, random_state=42)
    clf4 = RandomForestClassifier(n_estimators=100, random_state=42)
    clf5 = GradientBoostingClassifier(random_state=42)
    clf6 = AdaBoostClassifier(random_state=42)

    voting_clf = VotingClassifier(
        estimators=[('lr', clf1), ('knn', clf2), ('svc', clf3),
                    ('rf', clf4), ('gb', clf5), ('ab', clf6)],
        voting='soft'
    )

    voting_clf.fit(X_train, y_train)

    
    y_pred = voting_clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Voting Ensemble Accuracy: {acc:.2f}")

    joblib.dump(voting_clf, "voting_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("Model and scaler saved successfully!")


if __name__ == "__main__":
    train_voting_ensemble()
