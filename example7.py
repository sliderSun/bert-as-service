import random

from service.client import BertClient

num_sample = 10000
twitter_data = '/data/cips/data/lab/data/dataset/training.1600000.processed.noemoticon.csv'

print('loading...')
with open(twitter_data, 'r', encoding='utf8', errors='ignore') as fp:
    tmp = [v.split(',') for v in fp]
    dataset = [(int(v[0].strip().strip('"')), v[-1].strip().strip('"').strip()) for v in tmp]
    dataset = [v for v in dataset if v[1]]

print('%d samples loaded' % len(dataset))

subset = random.sample(dataset, num_sample)
subset_text = [v[1] for v in subset]
print('min_seq_len: %d' % min(len(v) for v in subset_text))
print('max_seq_len: %d' % max(len(v) for v in subset_text))

exit()
bc = BertClient(port=6000, port_out=6001)
subset_vec = bc.encode(subset_text)
