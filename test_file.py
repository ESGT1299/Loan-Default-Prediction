import pandas as pd
data = pd.read_csv("dataset/cleaned_data.csv")
data_sampled = data.sample(frac=0.2, random_state=42)

print(data_sampled.nunique())
