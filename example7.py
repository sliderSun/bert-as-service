import random

from MulticoreTSNE import MulticoreTSNE as TSNE
from matplotlib import pyplot as plt

from service.client import BertClient

num_sample = 1000
twitter_data = '/data/cips/data/lab/data/dataset/training.1600000.processed.noemoticon.csv'

print('loading...')
with open(twitter_data, 'r', encoding='utf8', errors='ignore') as fp:
    tmp = [v.split(',') for v in fp]
    dataset = [(int(v[0].strip().strip('"')), v[-1].strip().strip('"').strip()) for v in tmp]
    dataset = [v for v in dataset if v[1]]

print('%d samples loaded' % len(dataset))

subset = random.sample(dataset, num_sample)
subset_text = [v[1] for v in subset]
subset_label = [v[0] for v in subset]
print('min_seq_len: %d' % min(len(v.split()) for v in subset_text))
print('max_seq_len: %d' % max(len(v.split()) for v in subset_text))

bc = BertClient(port=6000, port_out=6001)
subset_vec = bc.encode(subset_text)
embeddings = TSNE(n_jobs=4).fit_transform(subset_vec)
vis_x = embeddings[:, 0]
vis_y = embeddings[:, 1]
plt.scatter(vis_x, vis_y, c=subset_label, cmap=plt.cm.get_cmap("jet", 10), marker='.')
plt.colorbar(ticks=range(len(subset_label)))
