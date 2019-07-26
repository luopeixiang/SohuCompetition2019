"""负责构建word2id, tag2id 以及根据word2id, tag2id将文本转化为训练文件"""

import random
import os
from os.path import join
import json
import multiprocessing
from functools import partial
from collections import Counter

import pkuseg
from tqdm import tqdm
import numpy as np

from data.processing import generate_char_tags, generate_word_tags, \
    generate_char_tags_partial
from data.building import extend_tag2id, extend_token2id


def split_data(data_dir, dev_num=500):
    """划分验证集、训练集"""
    total_file = join(data_dir, 'coreEntityEmotion_train.txt')
    train_file = join(data_dir, 'train_char.txt')
    dev_file = join(data_dir, 'dev_char.txt')

    with open(total_file) as total:
        lines = total.readlines()

    random.shuffle(lines)
    with open(dev_file, 'w') as dev:
        for line in lines[:dev_num]:
            dev.write(line)
    with open(train_file, 'w') as train:
        for line in lines[dev_num:]:
            train.write(line)
    print("Spliting Done!")


def segment(data_dir):

    print("Segmenting dev data...")
    dev_char = join(data_dir, 'dev_char.txt')
    dev_word = join(data_dir, 'dev_word.txt')
    segment_file(dev_char, dev_word)

    print("Segmenting training data...")
    train_char = join(data_dir, 'train_char.txt')
    train_word = join(data_dir, 'train_word.txt')
    segment_file(train_char, train_word)

    print("Segmenting test data...")
    test_char = join(data_dir, 'test_char.txt')
    src_name = join(data_dir, 'coreEntityEmotion_test_stage1.txt')
    if os.path.isfile(src_name):  # 重命名
        os.rename(src_name, test_char)
    test_word = join(data_dir, 'test_word.txt')
    segment_file(test_char, test_word)


def segment_file(src_file, tgt_file):
    seg = pkuseg.pkuseg()
    with open(src_file) as src:
        all_json_data = [json.loads(line) for line in src]

    with multiprocessing.Pool(processes=8) as pool:
        segmented = list(pool.map(
            partial(segment_single_item, seg),
            all_json_data,
            chunksize=1024
        ))

    with open(tgt_file, 'w') as tgt:
        for s in segmented:
            tgt.write(json.dumps(s, ensure_ascii=False)+'\n')


def segment_single_item(seg, json_data):
    json_data['title'] = seg.cut(json_data['title'])
    json_data['content'] = seg.cut(json_data['content'])

    return json_data


def build_token2id(data_dir, method, min_count=0):
    train_file = join(data_dir, 'train_{}.txt'.format(method))
    dev_file = join(data_dir, 'dev_{}.txt'.format(method))
    test_file = join(data_dir, 'test_{}.txt'.format(method))

    vocab = Counter()
    print("Building Vocab...")
    vocab.update(get_tokens(train_file))
    vocab.update(get_tokens(dev_file))
    vocab.update(get_tokens(test_file))

    # 保存token count
    token2id = {}
    token_count_file = join(data_dir, '{}_count.txt'.format(method))
    with open(token_count_file, 'w') as token_count:
        for token, count in vocab.most_common():
            token_count.write(token+'\t'+str(count)+'\n')
            if count > min_count:  # 丢弃只出现min_count次的token
                token2id[token] = len(token2id)

    # 保存token2id
    token2id = extend_token2id(token2id)

    json.dump(
        token2id,
        open(join(data_dir, '{}2id.json'.format(method)), 'w'),
        indent=4,
        ensure_ascii=False
    )
    print("Done!")


def get_tokens(articles_file):
    tokens = []
    with open(articles_file) as f:
        for line in f:
            json_data = json.loads(line)
            tokens += list(json_data['title'])
            tokens += list(json_data['content'])
    return tokens


def read_articles(data_file):
    articles = []
    with open(data_file) as f:
        for line in f:
            json_data = json.loads(line.strip())
            article = json_data['title'] + '。' + json_data['content']
            articles.append(article)
    return articles


def build_tag2id(save_dir, schema):
    assert schema in ['singleO', 'multiO']

    if schema == "singleO":
        tag2id = {
            "O": 0, "NORM_B": 1, "NORM_I": 2,
            "NORM_E": 3, "POS_B": 4, "POS_I": 5,
            "POS_E": 6, "NEG_B": 7, "NEG_E": 8,
            "NEG_I": 9, "POS_S": 10, "NEG_S": 11, "NORM_S": 12
        }
    elif schema == "multiO":
        tag2id = {
            "NORM_B": 0, "NORM_I": 1, "NORM_E": 2, "NORM_O": 3,
            "NA_O": 4, "POS_B": 5, "POS_I": 6, "POS_E": 7, "POS_O": 8,
            "NEG_B": 9, "NEG_I": 10, "NEG_E": 11, "NEG_O": 12,
            "POS_S": 13, "NEG_S": 14, "NORM_S": 15
        }

    tag2id = extend_tag2id(tag2id)
    json.dump(
        tag2id,
        open(join(save_dir, schema+'_tag2id.json'), 'w'),
        indent=4,
        ensure_ascii=False
    )


def building(data_dir, method, tag_schema='singleO'):
    """
    构建数据
    src_dir:数据所在的目录
    tgt_dir: 处理之后保存的目录
    method: word或者是char
    tag_schema: singleO或者是 multiO,先支持singleO(multiO好像并没有什么卵用)
    """

    ori_data_dir = join(data_dir, 'original_data')
    token2id_file = join(ori_data_dir, "{}2id.json".format(method))
    token2id = json.load(open(token2id_file))
    tag2id_file = join(ori_data_dir, "{}_tag2id.json".format(tag_schema))
    tag2id = json.load(open(tag2id_file))

    # 加载原始数据
    src_train = join(ori_data_dir, 'train_{}.txt'.format(method))
    src_dev = join(ori_data_dir, 'dev_{}.txt'.format(method))

    tgt_dir = join(data_dir, "{}_{}".format(method, tag_schema))
    if not os.path.isdir(tgt_dir):
        os.mkdir(tgt_dir)
    tgt_train = join(tgt_dir, 'train')
    tgt_dev = join(tgt_dir, 'dev')

    print("Converting dev data...")
    convert2id(src_dev, tgt_dev, token2id, tag2id, method)
    print("Converting training data...")
    convert2id(src_train, tgt_train, token2id, tag2id, method)
    print("Done!")


def convert2id(src, tgt, token2id, tag2id, method):
    assert method in ['char', 'word']
    articles_ids = []
    articles_tagsIds = []
    with open(src) as s:
        for line in tqdm(s):
            json_data = json.loads(line.strip())

            if method == "char":
                article, tags = generate_char_tags(json_data)
            elif method == "word":
                article, tags = generate_word_tags(json_data)

            unk = token2id['<unk>']
            article_ids = [token2id.get(token, unk) for token in article]
            tagsIds = [tag2id[tag] for tag in tags]
            articles_ids.append(article_ids)
            articles_tagsIds.append(tagsIds)

    np.savez(tgt, articles=articles_ids, tags=articles_tagsIds)


def building_partial(data_dir, method, sent_num=1, tag_schema='singleO'):
    """"与building功能类似，但只转化文章的前几句
    sent_num = 1时只转化title
    """
    ori_data_dir = join(data_dir, 'original_data')
    token2id_file = join(ori_data_dir, "{}2id.json".format(method))
    token2id = json.load(open(token2id_file))
    tag2id_file = join(ori_data_dir, "{}_tag2id.json".format(tag_schema))
    tag2id = json.load(open(tag2id_file))

    # 加载原始数据
    src_train = join(ori_data_dir, 'train_{}.txt'.format(method))
    src_dev = join(ori_data_dir, 'dev_{}.txt'.format(method))

    tgt_dir = join(data_dir, "{}_{}_partial{}".format(
        method, tag_schema, sent_num))
    if not os.path.isdir(tgt_dir):
        os.mkdir(tgt_dir)
    tgt_train = join(tgt_dir, 'train')
    tgt_dev = join(tgt_dir, 'dev')

    print("Converting dev data...")
    convert2id_partial(src_dev, tgt_dev, token2id,
                       tag2id, method, sent_num=sent_num)
    print("Converting training data...")
    convert2id_partial(src_train, tgt_train, token2id,
                       tag2id, method, sent_num=sent_num)
    print("Done!")


def convert2id_partial(src, tgt, token2id, tag2id, method, sent_num=1):
    assert method in ['char', 'word']
    articles_ids = []
    articles_tagsIds = []
    with open(src) as s:
        for line in tqdm(s):
            json_data = json.loads(line.strip())

            if method == "char":
                article, tags = generate_char_tags_partial(json_data, sent_num)
            elif method == "word":
                article, tags = generate_word_tags(json_data)  # 先留着

            unk = token2id['<unk>']
            article_ids = [token2id.get(token, unk) for token in article]
            tagsIds = [tag2id[tag] for tag in tags]
            articles_ids.append(article_ids)
            articles_tagsIds.append(tagsIds)

    np.savez(tgt, articles=articles_ids, tags=articles_tagsIds)


if __name__ == "__main__":
    ori_data_dir = "SohuData/original_data"
    split_data(ori_data_dir)

    # 分词
    segment(ori_data_dir)

    # 构建char2id, word2id
    build_token2id(ori_data_dir, 'char')
    build_token2id(ori_data_dir, 'word', min_count=1)

    # 构建tag2id
    build_tag2id(ori_data_dir, 'singleO')
    build_tag2id(ori_data_dir, 'multiO')

    # 将文本数据转化为数字表示

    root_dir = "SohuData"
    building(root_dir, 'char', 'singleO')
    building(root_dir, 'word', 'singleO')

    building_partial(root_dir, 'char', 1, 'singleO')
