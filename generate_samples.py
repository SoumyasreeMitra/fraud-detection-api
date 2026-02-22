import pandas as pd
import json

df = pd.read_csv("data/creditcard.csv")

# Get actual fraud row
fraud_sample = df[df["Class"] == 1].iloc[0].drop("Class").to_dict()

print(json.dumps(fraud_sample, indent=2))