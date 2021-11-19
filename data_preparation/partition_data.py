import pandas as pd
import csv

df = pd.read_csv("NBAData.csv")
# print(df)
dict_teams = {}
for index, d in df.iterrows():
    # print(d)
    if d["Team"] not in dict_teams:
        dict_teams[d["Team"]] = [1, [d["Tweet"]]]
    else:
        if dict_teams[d["Team"]][0] < 40:
            dict_teams[d["Team"]][0] += 1
            dict_teams[d["Team"]][1].append(d["Tweet"])
        else: 
            continue
# for d in dict_teams:
#     print(dict_teams[d][0])
dictt = []
for a in dict_teams:
    for i in range(40):
        dictt.append({"team": a, "tweet": dict_teams[a][1][i]})

csv_columns = ['team', 'tweet']
csv_file = "partitioned.csv"
try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in dictt:
            writer.writerow(data)
except IOError:
    print("I/O error")

