import pandas as pd
import json

df = pd.read_csv('rsc15-raw/rsc15-clicks.dat', names=["session_id", "timestamp", "item_id", "category"],
                         header=None, sep=',', engine='python').drop_duplicates(subset=['session_id', 'item_id'], keep='last')


items = list(df["item_id"].drop_duplicates())
index_session = {}
for item in items:
    index_session[item]=list(df[df["item_id"]==item]["session_id"])

with open('index_sknn.txt', 'w') as outfile:
    json.dump(index_session, outfile)