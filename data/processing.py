"""数据预处理"""

import re
import json
import random
import string
from collections import Counter

from config import TrainingConfig

# SENT_NUM = TrainingConfig.sentence_num


# def process(src_file, tgt_file, multi_O=False, sentence_num=SENT_NUM):

#     # 加载json数据
#     with open(src_file) as src:
#         with open(tgt_file, 'w') as tgt:
#             for line in src:
#                 line = line.strip('\n')
#                 json_data = json.loads(line)
#                 if multi_O:
#                     tokens, tags = generate_tags_multiO(
#                         json_data, sentence_num=sentence_num)
#                 else:
#                     tokens, tags = generate_char_tags(json_data)
#                 assert len(tokens) == len(tags)

#                 # 写入
#                 for token, tag in zip(tokens, tags):
#                     tgt.write(token + '\t' + tag + '\n')
#                 # 文章之间用空行隔开
#                 tgt.write('\n')


def generate_char_tags(json_data):
    """为char 生成标记"""
    article = json_data['title'] + '。' + json_data['content']
    entity_emotions = json_data["coreEntityEmotions"]

    article = clean(article)
    tags = ['O'] * len(article)

    entity_emotions.sort(
        key=lambda item: len(item['entity']),
        reverse=True
    )

    for item in entity_emotions:
        entity, emotion = item['entity'], item['emotion']
        entity = entity.strip()
        for start, end in find_all(entity, article):
            if check_safe_tagging(tags[start:end]):
                tags[start:end] = get_BIE_tags(
                    emotion,
                    len(entity)
                )

    return article, tags


def generate_char_tags_partial(json_data, sent_num):
    article = json_data['title'] + '。' + json_data['content']
    entity_emotions = json_data["coreEntityEmotions"]
    article = "".join(cut_sent(article)[:sent_num])

    article = clean(article)
    tags = ['O'] * len(article)

    entity_emotions.sort(
        key=lambda item: len(item['entity']),
        reverse=True
    )

    for item in entity_emotions:
        entity, emotion = item['entity'], item['emotion']
        entity = entity.strip()
        for start, end in find_all(entity, article):
            if check_safe_tagging(tags[start:end]):
                tags[start:end] = get_BIE_tags(
                    emotion,
                    len(entity)
                )
    return article, tags


def generate_word_tags(json_data):
    """为word生成标记"""
    # 先使用seg进行分词
    tokens = json_data['title'] + ['。'] + json_data['content']
    entity_emotions = json_data["coreEntityEmotions"]

    article = "".join(tokens)
    tags = ['O'] * len(tokens)
    token_lengths = [len(token) for token in tokens]
    token_span = [sum(token_lengths[:i]) for i in range(len(token_lengths)+1)]

    entity_emotions.sort(
        key=lambda item: len(item['entity']),
        reverse=True
    )  # 按照实体长度进行排序

    # 生成标记
    for item in entity_emotions:
        entity, emotion = item['entity'], item['emotion']
        entity = entity.lower().strip()
        positions = find_all(entity, article)
        for start, end in positions:
            try:
                token_start_ind = token_span.index(start)
                # 找end_ind
                token_end_ind = token_start_ind + 1
                while token_span[token_end_ind] < end:
                    token_end_ind += 1
            except ValueError:
                continue

            # 确认是安全的标记
            if check_safe_tagging(tags[token_start_ind:token_end_ind]):
                tags[token_start_ind:token_end_ind] = get_BIE_tags(
                    emotion,
                    token_end_ind-token_start_ind
                )
    assert len(tokens) == len(tags)
    return tokens, tags


def check_safe_tagging(tags):
    for tag in tags:
        if not tag.endswith('O'):  # 只要有一个标记不为O
            return False
    return True


def generate_tags_multiO(json_data, sentence_num=4):
    """采用第二种标记方案"""
    title = "".join(token for token in json_data['title'] if token != '\t')
    sentences = [title] + cut_sent(json_data['content'])
    sentences = sentences[:sentence_num]
    core_entities = json_data['coreEntityEmotions']
    sent_tags = generate_sent_tags(sentences, core_entities)

    article = list("".join(sentences))
    tags = []
    for sent_tag in sent_tags:
        tags += sent_tag
    return article, tags


def generate_sent_tags(sentences, core_entities):
    sent_tags = []
    for sent in sentences:
        sent_tag = []
        core_infos = []

        # 找到实体在句子中的位置
        core_entities.sort(key=lambda item: len(item['entity']), reverse=True)
        finded_entities = set()
        include = False
        for item in core_entities:
            entity, emotion = item['entity'], item['emotion']
            positions = find_all(entity, sent)

            for finded_e in finded_entities:  # 如果当前实体是已添加实体中某个实体的一部分，那么丢弃
                if entity in finded_e:
                    include = True

            if include:
                continue

            for pos in positions:
                info = {'entity': entity, 'emotion': emotion, 'pos': pos}
                core_infos.append(info)
                finded_entities.add(entity)

        # 若该句子中没有核心实体
        if len(core_infos) == 0:
            sent_tag = ['NA_O'] * len(sent)
            sent_tags.append(sent_tag)
            continue

        # 否则根据核心实体位置生成标签 首先从小到大进行排序
        core_infos.sort(key=lambda item: item['pos'][0])
        # 消除实体重叠的情况
        clean_core_infos = []
        overlap = False
        for i, core_info in enumerate(core_infos):
            start, end = core_info['pos']
            for other_core_info in core_infos[i+1:]:
                other_start, _ = other_core_info['pos']
                if other_start < end:
                    overlap = True
                    break
            if not overlap:
                clean_core_infos.append(core_info)
            overlap = False

        entity_ind = 0
        ind = 0
        while ind < len(sent):
            try:
                start, end = clean_core_infos[entity_ind]['pos']
                cur_emotion = clean_core_infos[entity_ind]['emotion']
                cur_entity = clean_core_infos[entity_ind]['entity']
            except IndexError:  # 最后一个实体 之后的标注
                sent_tag += [cur_emotion+'_O']*(len(sent)-end)
                break

            while ind < start:
                sent_tag.append(cur_emotion+'_O')
                ind += 1
            sent_tag += get_BIE_tags(cur_emotion, len(cur_entity))
            ind += len(cur_entity)
            entity_ind += 1

        if len(sent_tag) != len(sent):
            import pdb
            pdb.set_trace()
        sent_tags.append(sent_tag)

    return sent_tags


def cut_sent(para):
    para = "".join([token for token in para if token != '\t'])
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    para = [p for p in para.split("\n") if p]
    return para


def find_all(e, article):

    results = []
    index = 0
    end_find = False
    while not end_find:
        start = article.find(e, index)
        if start != -1:
            index = start+len(e)
            results.append((start, index))
        else:
            end_find = True

    return results


def get_BIE_tags(emotion, length):

    if length == 1:
        return [emotion+'_S']
    elif length == 2:
        return [emotion+'_B', emotion+'_E']
    else:
        return [emotion+'_B'] + [emotion+'_I'] * (length-2) + [emotion+'_E']


def clean(article):
    """消除\t \n等空白字符（但保留空格）"""
    dirty_chars = string.whitespace.replace(" ", "")
    article = [token for token in article if token not in dirty_chars]
    article = "".join(article)
    return article


def load_data(tag_file):
    """从标记文件中加载句子及其对应的标记"""
    sents = []
    total_tags = []

    with open(tag_file) as f:
        sent = []
        sent_tags = []
        for line in f:
            line = line.strip('\n')
            if line:
                try:
                    token, tag = line.split('\t')
                except ValueError:
                    import pdb
                    pdb.set_trace()
                sent.append(token)
                sent_tags.append(tag)
            else:
                sents.append(sent)
                total_tags.append(sent_tags)

                sent = []
                sent_tags = []

    return sents, total_tags


def split_train_and_dev(total, train, dev, dev_num=1000):
    sents, tags = load_data(total)

    pairs = list(zip(sents, tags))
    random.shuffle(pairs)

    print("写入验证集...")
    with open(dev, 'w') as dev_file:
        for p in pairs[:dev_num]:
            for token, tag in zip(*p):
                dev_file.write(token + '\t' + tag + '\n')
            dev_file.write('\n')

    print("写入训练集...")
    with open(train, 'w') as train_file:
        for p in pairs[dev_num:]:
            for token, tag in zip(*p):
                train_file.write(token + '\t' + tag + '\n')
            train_file.write('\n')


def get_entity_emotion(sent: [str], tags: [str], all_return=False):
    """给定一个句子，以及对应的标记，抽取出实体，以及情感
        all: 若为True则返回所有提取出的实体，若为False则返回出现次数前3的实体
    """
    assert len(sent) == len(tags)
    entities = []

    entity = ""
    for word, tag in zip(sent, tags):
        if tag.endswith('_S'):  # 单实体
            emotion = tag[:-2]
            entities.append(word+"_"+emotion)
        elif tag.endswith('_B') or tag.endswith('_I'):  # 实体内部
            entity += word
        elif tag.endswith('_E'):  # 实体结束字段
            entity += word
            emotion = tag[:-2]
            entities.append(entity+"_"+emotion)
            entity = ""
        else:  # O标记
            continue

    # 进一步处理
    entities, emotions = process_results(entities, all_return=all_return)
    return entities, emotions


def process_results(entities, all_return):
    c = Counter(entities)
    entity_results = []
    emotion_results = []

    for entity_emotion, count in c.most_common():
        try:
            entity, emotion = entity_emotion.split('_')
        except ValueError:
            continue
        if entity not in entity_results:
            entity_results.append(entity)
            emotion_results.append(emotion)

            if not all_return:
                if len(entity_results) == 3:
                    break
    assert len(entity_results) == len(emotion_results)

    return entity_results, emotion_results


if __name__ == "__main__":
    # src_file = './SohuData/original_data/train.txt'
    # tgt_file = './SohuData/multiO_tagged_data/total.txt'

    # print("正在标记文件....")
    # process(src_file, tgt_file, multi_O=True, sentence_num=SENT_NUM)

    # # 划分验证集、训练集， 验证集取1000条数据，训练集是39000
    # print("划分训练集，验证集....")
    # train_file = './SohuData/multiO_tagged_data/train.txt'
    # dev_file = './SohuData/multiO_tagged_data/dev.txt'
    # split_train_and_dev(tgt_file, train_file, dev_file, dev_num=500)

    pass
