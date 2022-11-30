import re
import sys

import joblib
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine

nltk.download("omw-1.4")
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()


def load_data(database_filepath):
    """
    Load dataset from SQL database, split into data and label
    Input: SQL database file path
    Output: data X, label Y and label name
    """
    engine = create_engine("sqlite:///{}".format(database_filepath))
    df = pd.read_sql_table("disaster_table", engine)
    X = df["message"]
    Y = df.drop(["id", "message", "original", "genre"], axis=1)

    return X, Y, Y.columns


def tokenize(text):
    """
    Tokenizer text
    Input: a string text
    Output: tokenizer of text
    """
    # normalization
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)

    # tokenize
    token = word_tokenize(text)

    # Remove stop words
    words = [word for word in token if word not in stop_words]
    lemmed = [lemmatizer.lemmatize(w) for w in words]

    return lemmed


def build_model():
    """
    Build machine learning model with pipeline and optimize parameters
    """
    pipeline = Pipeline(
        [
            ("vect", CountVectorizer(tokenizer=tokenize)),
            ("tfidf", TfidfTransformer()),
            (
                "classifier",
                MultiOutputClassifier(RandomForestClassifier(n_estimators=10)),
            ),
        ]
    )

    parameters = {
        "tfidf__use_idf": [True, False],
        "classifier__estimator__criterion": ["gini", "entropy"],
        "classifier__estimator__n_estimators": [10, 100],
    }

    model = GridSearchCV(pipeline, param_grid=parameters)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model with f1 score, precision and recall metrics by category in 36 categories
    Input:
        - Trained model
        - Data X_test
        - Label Y_test
        - Label name category_names
    """
    Y_pred = model.best_estimator_.predict(X_test)

    for i, cate in enumerate(category_names):
        print("-" * 20)
        print("Category:", cate.capitalize())
        print(classification_report(Y_test.iloc[:, i], Y_pred[:, i]))


def save_model(model, model_filepath):
    """
    Save trained model into a weight file
    Input:
        - Trained model
        - Destination path of model file
    """
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print("Loading data...\n    DATABASE: {}".format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print("Building model...")
        model = build_model()

        print("Training model...")
        model.fit(X_train, Y_train)

        print("Evaluating model...")
        evaluate_model(model, X_test, Y_test, category_names)

        print("Saving model...\n    MODEL: {}".format(model_filepath))
        save_model(model, model_filepath)

        print("Trained model saved!")

    else:
        print(
            "Please provide the filepath of the disaster messages database "
            "as the first argument and the filepath of the pickle file to "
            "save the model to as the second argument. \n\nExample: python "
            "train_classifier.py ../data/DisasterResponse.db classifier.pkl"
        )


if __name__ == "__main__":
    main()
