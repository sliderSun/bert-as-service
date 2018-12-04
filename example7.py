twitter_data = '/data/cips/data/lab/data/dataset/training.1600000.processed.noemoticon.csv'

print('loading...')
with open(twitter_data, 'r', encoding='utf8', errors='ignore') as fp:
    tmp = [v.split(',') for v in fp]
    dataset = [(v[0].strip().strip('"'), v[-1].strip().strip('"')) for v in tmp]

print('%d samples loaded' % len(dataset))
