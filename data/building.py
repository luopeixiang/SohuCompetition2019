"""用于构建数据"""

from os.path import join
import json

from config import TrainingConfig


# # const
# WORD2ID_FILE = TrainingConfig.word2id_file
# TAG2ID_FILE = TrainingConfig.tag2id_file
# F_LEN = TrainingConfig.forward_len
# B_LEN = TrainingConfig.backward_len
# DATA_DIR = TrainingConfig.data_dir


# def build_corpus(split,
#                  make_vocab=True,
#                  data_dir=DATA_DIR):
#     """读取数据"""
#     assert split in ['train', 'dev', 'test']

#     word_lists = []
#     tag_lists = []
#     with open(join(data_dir, split+".txt"), 'r') as f:
#         word_list = []
#         tag_list = []
#         for line in f:
#             if line != '\n':
#                 word, tag = line.strip('\n').split('\t')
#                 word_list.append(word)
#                 tag_list.append(tag)
#             else:
#                 word_lists.append(word_list)
#                 tag_lists.append(tag_list)
#                 word_list = []
#                 tag_list = []

#     # 将过长的文章截断,加快训练
#     word_lists = [truncate_article(word_list) for word_list in word_lists]
#     tag_lists = [truncate_article(tag_list) for tag_list in tag_lists]

#     # 如果make_vocab为True，还需要返回word2id和tag2id
#     if make_vocab:
#         word2id = build_map(word_lists, out_file=WORD2ID_FILE)
#         tag2id = build_map(tag_lists, out_file=TAG2ID_FILE)
#         return word_lists, tag_lists, word2id, tag2id
#     else:
#         return word_lists, tag_lists


# def truncate_article(article, f_len=F_LEN, b_len=B_LEN):
#     assert f_len != 0 or b_len != 0  # 二者不能同时为0

#     if len(article) < f_len + b_len:
#         return article

#     if f_len == 0 and b_len != 0:  # 只取后半段
#         article = article[-b_len:]
#     elif f_len != 0 and b_len == 0:
#         article = article[:f_len]
#     else:
#         article = article[:f_len] + article[-b_len:]
#     return article


# def build_map(lists, out_file=None):
#     maps = {}
#     for list_ in lists:
#         for e in list_:
#             if e not in maps:
#                 maps[e] = len(maps)

#     if out_file:
#         with open(out_file, 'w') as out:
#             out.write(json.dumps(maps, indent=4, ensure_ascii=False))

#     return maps

# # LSTM模型训练的时候需要在word2id和tag2id加入PAD和UNK
# # 如果是加了CRF的lstm还要加入<start>和<end> (解码的时候需要用到)


# def extend_maps(word2id, tag2id, for_crf=True):
#     word2id['<unk>'] = len(word2id)
#     word2id['<pad>'] = len(word2id)
#     tag2id['<unk>'] = len(tag2id)  # ?
#     tag2id['<pad>'] = len(tag2id)

#     # 如果是加了CRF的bilstm  那么还要加入<start> 和 <end>token
#     if for_crf:
#         word2id['<start>'] = len(word2id)
#         word2id['<end>'] = len(word2id)
#         tag2id['<start>'] = len(tag2id)
#         tag2id['<end>'] = len(tag2id)

#     return word2id, tag2id


def extend_token2id(word2id):
    word2id['<unk>'] = len(word2id)
    word2id['<pad>'] = len(word2id)
    word2id['<start>'] = len(word2id)
    word2id['<end>'] = len(word2id)
    return word2id


def extend_tag2id(tag2id):
    tag2id['<pad>'] = len(tag2id)
    tag2id['<start>'] = len(tag2id)
    tag2id['<end>'] = len(tag2id)

    return tag2id


def prepocess_data_for_lstmcrf(word_lists, tag_lists, test=False):
    assert len(word_lists) == len(tag_lists)
    for i in range(len(word_lists)):
        word_lists[i].append("<end>")
        if not test:  # 如果是测试数据，就不需要加end token了
            tag_lists[i].append("<end>")

    return word_lists, tag_lists


def merge_maps(dict1, dict2):
    """用于合并两个word2id或者两个tag2id"""
    for key in dict2.keys():
        if key not in dict1:
            dict1[key] = len(dict1)
    return dict1
