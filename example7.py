twitter_data = '/data/cips/data/lab/data/dataset/training.1600000.processed.noemoticon.csv'

with open(twitter_data, 'r', encoding='utf8') as fp:
    tmp = [v.split(',') for v in fp]
    tmp = [(v[0].strip('"'), v[-1].strip('"')) for v in tmp]

print(tmp[-10:])
print(tmp[:10])
