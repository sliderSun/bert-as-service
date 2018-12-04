twitter_data = '/data/cips/data/lab/data/dataset/training.1600000.processed.noemoticon.csv'

print('loading...')
with open(twitter_data, 'r', encoding='utf8', errors='ignore') as fp:
    tmp = [v.split(',') for v in fp]
    dataset = [(int(v[0].strip('\s"')), v[-1].strip('"\n\s')) for v in tmp]

print('%d samples loaded' % len(dataset))
print(dataset[:10])
print(dataset[-10:])
