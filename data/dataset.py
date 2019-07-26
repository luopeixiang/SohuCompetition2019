from os.path import join
import json

import numpy as np
import torch
from torch.utils.data import Dataset

from config import TrainingConfig
from data.processing import clean, cut_sent


class SohuDataSet(Dataset):
    """负责训练集、验证集的加载"""

    def __init__(self, split: str, path: str) -> None:
        assert split in ['train', 'dev']

        data = np.load(join(path, split+'.npz'))
        self.articles = data['articles']
        self.tags = data['tags']

    def __len__(self):
        return len(self.tags)

    def __getitem__(self, index):
        return self.articles[index], self.tags[index]


class SohuTestDataSet(Dataset):
    """负责测试集的加载"""

    def __init__(self, data_dir, split, method):
        self.path = join(data_dir, '{}_{}.txt'.format(split, method))
        self.method = method

        self.article_ids = []
        self.articles = []
        self._load_data()

    def __len__(self):
        return len(self.article_ids)

    def __getitem__(self, index):
        return self.article_ids[index], self.articles[index]

    def _load_data(self):
        with open(self.path) as f:
            for line in f:
                json_data = json.loads(line)

                self.article_ids.append(json_data['newsId'])
                if self.method == 'char':
                    article = json_data['title'] + '。' + json_data['content']
                    article = list(clean(article))
                else:
                    article = json_data['title'] + ['。'] + json_data['content']
                self.articles.append(article)


def collate_tagged(token2id, tag2id, method, max_len, batch):

    batch.sort(key=lambda item: len(item[0]), reverse=True)
    articles, tags = zip(*batch)

    batch_size = len(articles)
    tag_pad = tag2id['<pad>']
    token_pad = token2id['<pad>']

    # 较长的文章进行截断
    articles = [article[:max_len] for article in articles]
    tags = [tag[:max_len] for tag in tags]

    # 如果是使用的方法是条件随机场，那么还需要加上end token
    if method == "lstm_crf":
        end_token = token2id['<end>']
        end_tag = tag2id['<end>']
        for i in range(len(articles)):
            articles[i].append(end_token)
            tags[i].append(end_tag)
        max_len += 1

    # 填充,对较短的数据进行填充
    articles_tensor = torch.ones(batch_size, max_len).long() * token_pad
    tags_tensor = torch.ones(batch_size, max_len).long() * tag_pad
    lengths = [len(art) for art in articles]
    for i, l in enumerate(lengths):
        articles_tensor[i][:l] = torch.LongTensor(articles[i])
        tags_tensor[i][:l] = torch.LongTensor(tags[i])

    return articles_tensor, tags_tensor, lengths


def collate_untagged(token2id, method, max_len, batch):

    # 根据文章长度进行排序
    batch.sort(key=lambda item: len(item[1]), reverse=True)
    article_ids, articles = zip(*batch)

    batch_size = len(articles)
    token_pad = token2id['<pad>']
    token_unk = token2id['<unk>']

    # 较长的文章进行截断
    articles = [article[:max_len] for article in articles]

    # 如果是使用的方法是条件随机场，那么还需要加上end token
    if method == "lstm_crf":
        end_token = token2id['<end>']
        for i in range(len(articles)):
            articles[i].append(end_token)
        max_len += 1

    # 填充,对较短的数据进行填充
    articles_tensor = torch.ones(batch_size, max_len).long() * token_pad
    lengths = [len(art) for art in articles]
    for i, l in enumerate(lengths):
        token_ids = [token2id.get(token, token_unk) for token in articles[i]]
        articles_tensor[i][:l] = torch.LongTensor(token_ids)

    return articles_tensor, article_ids, lengths
