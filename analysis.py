import pandas as pd
import os

path = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(path + "\\star_classification.csv")

print(df.head())