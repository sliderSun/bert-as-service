import random

import matplotlib

matplotlib.use('Agg')
from MulticoreTSNE import MulticoreTSNE as TSNE
from matplotlib import pyplot as plt
from matplotlib.pyplot import savefig

from service.client import BertClient

num_sample = 10000
twitter_data = '/data/cips/data/lab/data/dataset/training.1600000.processed.noemoticon.csv'

print('loading...')
with open(twitter_data, 'r', encoding='utf8', errors='ignore') as fp:
    tmp = [v.split(',') for v in fp]
    tmp = [(int(v[0].strip().strip('"')), v[-1].strip().strip('"').strip()) for v in tmp]
    dataset = [v for v in tmp if v[1]]

print('%d samples loaded' % len(dataset))

random.seed(531)
subset = random.sample(dataset, num_sample)
print(subset[:10])
print(subset[-10:])

subset_text = [v[1] for v in subset]
subset_label = [v[0] for v in subset]
num_label = len(set(subset_label))
print('min_seq_len: %d' % min(len(v.split()) for v in subset_text))
print('max_seq_len: %d' % max(len(v.split()) for v in subset_text))
print('unique label: %d' % num_label)

bc = BertClient(port=6000, port_out=6001)
subset_vec = bc.encode(subset_text)
embeddings = TSNE(n_jobs=8).fit_transform(subset_vec)
vis_x = embeddings[:, 0]
vis_y = embeddings[:, 1]
plt.scatter(vis_x, vis_y, c=subset_label, cmap=plt.cm.get_cmap("jet", num_label), marker='.')
plt.colorbar(ticks=range(num_label))
savefig('layer.png', bbox_inches='tight')
