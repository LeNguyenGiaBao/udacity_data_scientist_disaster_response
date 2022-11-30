import sys

import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load dataset: message data and category data
    Input: 2 file paths, with csv format
    Output: A dataframe was joined from message dataset and category dataset,
        by "id" column
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    return pd.merge(messages, categories, on="id")


def clean_data(df):
    """
    Process data
    Input: A dataset from 2 csv files, joined by id
    Output: A processed dataset
    Step:
        - Refactor category dataset with column name and numeric value
        - Drop the duplicates
    """
    categories = df["categories"].str.split(";", expand=True)
    category_colnames = categories.iloc[0].apply(lambda x: x.split("-")[0])
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split("-")[1])

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    df = df.drop("categories", axis=1)
    df = pd.concat([df, categories], axis=1, join="inner")
    df = df.drop(df[df["related"] == 2].index)
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    """
    Save dataset into SQL database file
    Input: dataframe and database filename
    """
    engine = create_engine("sqlite:///{}".format(database_filename))
    df.to_sql("disaster_table", engine, index=False, if_exists="replace")


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(
            "Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}".format(
                messages_filepath, categories_filepath
            )
        )
        df = load_data(messages_filepath, categories_filepath)

        print("Cleaning data...")
        df = clean_data(df)

        print("Saving data...\n    DATABASE: {}".format(database_filepath))
        save_data(df, database_filepath)

        print("Cleaned data saved to database!")

    else:
        print(
            "Please provide the filepaths of the messages and categories "
            "datasets as the first and second argument respectively, as "
            "well as the filepath of the database to save the cleaned data "
            "to as the third argument. \n\nExample: python process_data.py "
            "disaster_messages.csv disaster_categories.csv "
            "DisasterResponse.db"
        )


if __name__ == "__main__":
    main()
