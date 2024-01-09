import pandas as pd

#data = pd.read_csv('BobbleAI/valid1.csv')
data = pd.read_json('BobbleAI/hin_valid.json',lines=True)
print(data.head)