"""用于帮助分析数据的函数"""

import json
from collections import Counter

from data.processing import get_entity_emotion, clean, find_all
from utils.decoding import load_nerDict
from .evaluation import EntityEmotionMetrics


def analysis_result(data_loader, outfile, split='dev'):
    """
    分析结果
    主要是两个方面：
    1.准确率 召回率 f1, 混淆矩阵
    2.句子 目标实体， 以及 预测的实体 -->看看预测出来的实体都是些什么东西
    """
    pass


def analysis_data(json_file, save_file):
    """读取源数据文件，分析:
        每篇文章的长度，核心实体的出现次数，
        核心实体出现的位置分布
        用于帮助调参
    """
    out = open(save_file, 'w')
    with open(json_file) as f:
        info = {}
        for line in f:
            json_data = json.loads(line)
            info['news_id'] = json_data['newsId']
            info['entity_count'] = len(json_data['coreEntityEmotions'])

            article = json_data['title'] + '。' + json_data['content']
            article = clean(article)
            info['news_length'] = len(article)

            # 每个实体出现的位置
            positions = []
            for item in json_data['coreEntityEmotions']:
                entity = item['entity']
                entity_pos = find_all(entity, article)
                positions.append(entity_pos)
            info['positions'] = positions
            out.write(json.dumps(info, ensure_ascii=False)+'\n')
            info = {}
    out.close()


def count_entity_inTitle(json_file):
    """计算核心实体出现在title中的比例"""
    entity_count = 0
    entity_inTitle_count = 0
    with open(json_file) as f:
        for line in f:
            json_data = json.loads(line)
            title = json_data['title']
            for item in json_data['coreEntityEmotions']:
                if item['entity'] in title:
                    entity_inTitle_count += 1
                entity_count += 1
    return entity_count, entity_inTitle_count


def count_entity_inArticle(json_file):
    """计算核心实体只在文章出现过一次的比例"""
    entity_count = 0
    entity_one_count = 0
    with open(json_file) as f:
        for line in f:
            json_data = json.loads(line)
            article = json_data['title'] + json_data['content']
            for item in json_data['coreEntityEmotions']:
                if len(find_all(item['entity'], article)) < 2:
                    entity_one_count += 1
                entity_count += 1
    return entity_count, entity_one_count


def count_entity_inNerDict(json_file):
    """计算出现在NerDict中的实体的比例"""
    entity_count = 0
    entity_inNerDict = 0
    nerDict = load_nerDict()
    with open(json_file) as f:
        for line in f:
            json_data = json.loads(line)
            for item in json_data['coreEntityEmotions']:
                if item['entity'] in nerDict:
                    entity_inNerDict += 1
                entity_count += 1
    return entity_count, entity_inNerDict


def map_length(length):
    """将长度分类为:
        1.400以下
        2.400-600
        3.600-800
        4.800-1200
        5.1200以上
    """
    if length <= 400:
        return "400以下"
    elif length > 400 and length <= 600:
        return "400-600"
    elif length > 600 and length <= 800:
        return "600-800"
    elif length > 800 and length < 1200:
        return "800-1200"
    else:
        return "1200以上"


def get_heads(pos_pairs):
    heads = [start for start, end in pos_pairs]
    return heads


def load_infos(info_file):
    infos = []
    with open(info_file) as f:
        for line in f:
            infos.append(json.loads(line.strip()))
    return infos


def get_pos(infos, forward_len=400, back_len=400):
    """分析实体出现的位置
        1.出现在前100个词的占比=在前100个词中出现的次数/在整个文章中的总次数
        2.出现在后100个词的占比=在后100个词中出现的次数/在整个文章中的总次数 
        文章的长度越长，越会找出奇奇怪怪的实体,分析实体的位置，有助于确定取文章的哪些部分作为输入
    """

    count = 0  # 实体出现的总次数
    forward_count = 0
    backward_count = 0
    bf_count = 0
    bf_count_both = 0

    ecount = 0  # 总的实体个数
    eforward_count = 0  # 实体出现在前forwar_len个词，则加1
    ebackward_count = 0
    ebf_count = 0
    ebf_count_both = 0  # 同时出现在尾部 和前部

    for info in infos:
        length = info['news_length']
        ecount += info['entity_count']
        for position_list in info['positions']:
            heads = get_heads(position_list)
            if not heads:
                continue
            count += len(heads)
            for head in heads:
                if head < forward_len:
                    forward_count += 1
                if head > (length-back_len):
                    backward_count += 1
                if head < forward_len or head > (length-back_len):
                    bf_count += 1
                if head < forward_len and head > (length-back_len):
                    bf_count_both += 1

            if heads[0] < forward_len:
                eforward_count += 1
            if heads[-1] > (length-back_len):
                ebackward_count += 1

            if heads[0] < forward_len or heads[-1] > (length-back_len):
                ebf_count += 1
            if heads[0] < forward_len and heads[-1] > (length-back_len):
                ebf_count_both += 1

    out_format = "{:.2f}%\t" * 8
    out_format = """出现在前n个词的实体的比例（除以总次数）: {:.2f}%\n
                    出现在后n个词的实体的比例（次数）：{:.2f}%\n
                    前后面出现过一次的比例（衡量覆盖度):{:.2f}%\n
                    前后面都出现过的比例: {:.2f}%\n
                    在前n个词出现过的比例（除以实体总数）：{:.2f}%\n
                    在后n个词出现过的比例：{:.2f}%\n
                    前后面出现过一次的比例:{:.2f}%\n
                    前后面都出现过的比例:{:.2f}%\n
                """

    out = out_format.format(
        100 * forward_count / count,  # 出现在前n个词的实体的比例 次数
        100 * backward_count / count,  # 出现在后n个词的比例
        100 * bf_count / count,  # 前后面 出现过一次的比例(覆盖度)
        100 * bf_count_both / count,  # 前后面都出现过的比例
        100 * eforward_count / ecount,  # 在前n个词出现过的比例
        100 * ebackward_count / ecount,  # 在后n个词
        100 * ebf_count / ecount,
        100 * ebf_count_both / ecount
    )
    print(out)
    print("共有{}个核心实体，共出现了{}次".format(ecount, count))


def cal_tag_ratio(sent_tags, max_len):
    """分析每种标签所占的比例"""
    total_tags = []
    for t in sent_tags:
        total_tags += t[:max_len]

    tags_counter = Counter(total_tags)

    for tag, count in tags_counter.items():
        print('{} {:.2f}%'.format(tag, 100*count/len(total_tags)))


def analysis_entity_length(json_file):
    """统计实体的长度分布"""
    entities = []
    with open(json_file) as f:
        for line in f:
            json_data = json.loads(line)
            for item in json_data['coreEntityEmotions']:
                entities.append(item["entity"])

    length = [len(e) for e in entities]
    length_counter = Counter(length)

    return length_counter


if __name__ == "__main__":
    # result_file = "./tmp_lstm"
    # analysis_result(result_file)
    pass
