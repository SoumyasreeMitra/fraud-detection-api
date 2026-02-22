import pandas as pd

def load_and_preprocess(path):

    df = pd.read_csv(path)

    # Target column
    df.rename(columns={"Class": "Fraud"}, inplace=True)

    # Drop Time (not very useful initially)
    df.drop("Time", axis=1, inplace=True)

    return df