"""用于帮助解码的函数"""

import json
from os.path import join
import math
from functools import partial
import multiprocessing
from collections import Counter

from torch.utils.data import DataLoader

from data.processing import get_entity_emotion, clean, find_all, cut_sent
from data.dataset import SohuTestDataSet, collate_untagged
# from data.building import truncate_article, extend_maps
from config import TrainingConfig
from .model import load_model

# WORD2ID_FILE = TrainingConfig.word2id_file
# TAG2ID_FILE = TrainingConfig.tag2id_file
# TEST_JSON = "/home/luopx/Project/Sohu/SohuData/original_data/test_stage1.txt"
NER_DICT = "/home/luopx/Project/Sohu/SohuData/nerDict.txt"

# # 加载指定模型，在训练集上解码，获得结果


def load_and_decoding(model_path,
                      save_file,
                      use_model='best_val_loss'
                      ):

    assert use_model in ['best_val_loss', 'best_f1']
    print('Loading....')
    model = load_model(join(model_path, 'model.pkl'))
    # Remove Warning
    if model.method == 'lstm_crf':
        model.model.bilstm.bilstm.flatten_parameters()
    elif model.method == 'lstm':
        model.model.bilstm.flatten_parameters()
    elif model.method == 'lstm_lstm':
        model.model.encoder.flatten_parameters()

    meta_info = json.load(open(join(model_path, 'meta.json')))
    test_loader = DataLoader(
        SohuTestDataSet('SohuData/original_data', meta_info['token_method']),
        batch_size=TrainingConfig.decoding_batch_size,
        collate_fn=partial(collate_untagged, model.token2id,
                           meta_info['method'], meta_info['max_len'])
    )

    print("Decoding...")
    decoding(model, test_loader, save_file, use_model=use_model)


def decoding(model, test_loader, save_file, use_model="best_val_loss"):

    pred_tags, article_ids, articles_truncated = model.test(
        test_loader, use_model=use_model)

    print("将标注转化为实体，并将结果写到文件{}...".format(save_file))
    out_file = open(save_file, 'w')
    for art_truncated, art_id, tags in zip(articles_truncated, article_ids, pred_tags):
        try:
            entities, emotions = get_entity_emotion(
                art_truncated, tags, all_return=True)
        except ValueError:
            entities = []
            emotions = []
        # 按照格式写入到文件用于提交
        out_file.write(art_id+'\t')
        out_file.write(','.join(entities)+'\t')
        out_file.write(','.join(emotions)+'\n')

    print("写入结束!")
    out_file.close()


def load_complete_articles(data_dir="SohuData/original_data", split="test"):
    """加载完整的文章，用于筛选核心实体"""
    assert split in ['train', 'dev', 'test']
    data_file = join(data_dir, "{}_char.txt".format(split))
    articles = {}
    with open(data_file) as f:
        for line in f:
            json_data = json.loads(line)
            article = clean(json_data['title']+"。"+json_data['content'])
            articles[json_data['newsId']] = article
    return articles


def choose_core_entities(entities, emotions, article, nerDict):
    # 首先消除重复的实体
    all_pairs = [(entity, emotion)
                 for entity, emotion in zip(entities, emotions)]
    all_pairs = Counter(all_pairs)
    filtered_pairs = []
    keep_entities = []
    for (entity, emotion), count in all_pairs.most_common():
        if entity not in keep_entities and \
                entity in nerDict and len(entity) > 1:
            filtered_pairs.append((entity, emotion))
            keep_entities.append(entity)

    # 将是长实体一部分的 短实体去掉
    # 首先根据长度进行排序
    # filtered_pairs.sort(key=lambda p: len(p[0]))
    # overlap = False
    # entities = []
    # emotions = []
    # for i, (entity, emotion) in enumerate(filtered_pairs[:-1]):
    #     for o_entity, o_emotion in filtered_pairs[i+1:]:
    #         if entity in o_entity:
    #             overlap = True
    #     if not overlap:
    #         entities.append(entity)
    #         emotions.append(emotion)
    #     overlap = False

    # if len(filtered_pairs) > 0:
    #     entities.append(filtered_pairs[-1][0])
    #     emotions.append(filtered_pairs[-1][1])

    # 不将短实体去掉了
    if len(filtered_pairs) > 0:
        entities, emotions = list(zip(*filtered_pairs))
    else:
        entities = []
        emotions = []

    # 如果经过过滤之后的实体数还是比3大，则根据在原文的出现次数进行排序，返回前三个
    if len(entities) > 3:
        entities, emotions = choose_by_count(entities, emotions, article)

    return entities, emotions


def choose_by_count(entities, emotions, article):
    results = []
    for entity, emotion in zip(entities, emotions):
        count = len(find_all(entity, article))
        if count > 1:  # 只要出现次数大于1的
            results.append(
                (entity, emotion, count)
            )
    # 排序
    results.sort(key=lambda item: item[2], reverse=True)
    if len(results) > 0:
        entities, emotions, _ = list(zip(*results))

    return entities, emotions


def choose_byTfidf(entities, emotions, article, all_ariticles):
    """先根据实体词典过滤实体，然后根据实体的tf-idf值选择实体"""
    scores = []

    # for entity in entities:  # 计算scores  考虑换成多线程
    #     # 计算tf
    #     tf = len(find_all(entity, article)) / len(article)
    #     idf = len(all_ariticles) / \
    #         (sum(entity in art for art in all_ariticles) + 1)
    #     scores.append(tf * math.log(idf))

    with multiprocessing.Pool(processes=8) as pool:
        scores = list(pool.imap(
            partial(cal_tfidf, all_ariticles, article),
            entities
        ))

    entities = sorted(entities,
                      key=lambda e: scores[entities.index(e)], reverse=True)
    emotions = sorted(emotions,
                      key=lambda e: scores[emotions.index(e)], reverse=True)

    return entities[:3], emotions[:3]


def cal_tfidf(all_ariticles, article, entity):
    """计算某个实体的tfidf值"""
    tf = len(find_all(entity, article)) / len(article)
    idf = len(all_ariticles) / \
        (sum(entity in art for art in all_ariticles) + 1)
    tf_idf = tf * math.log(idf)
    return tf_idf


def load_nerDict():
    entities_set = set()
    with open(NER_DICT) as f:
        for line in f:
            entities_set.add(line.strip())
    return entities_set


def load_maps(path):
    with open(join(path, 'word2id.json')) as f:
        word2id = json.loads(f.read())

    with open(join(path, 'tag2id.json')) as f:
        tag2id = json.loads(f.read())

    return word2id, tag2id


# 帮助对比分析结果的函数
def get_results(test_sents, pred_tags, tgt_tags):

    for sent, tags, tgt_tags in zip(test_sents, pred_tags, tgt_tags):
        entities = get_entity_emotion(sent, tags)
        tgt_entities = get_entity_emotion(sent, tgt_tags)
    print("".join(sent))
    print(entities)
    print(Counter(entities))
    print("TARGET：", Counter(tgt_entities))
    print("="*50)
    print('\n')


def check_results(result):
    with open(result) as f:
        for line in f:
            try:
                id_, entities, emotions = line.strip().split('\t')
                entities = entities.split(',')
                emotions = emotions.split(',')
                assert len(emotions) == len(entities)
            except:
                print(line)
                import pdb
                pdb.set_trace()


def load_result(result_file):
    """加载结果"""
    result = {}
    with open(result_file) as f:
        for line in f:
            line = line.strip('\n')
            try:
                news_id, entities, emotions = line.split('\t')
                entities = entities.split(',')
                emotions = emotions.split(',')
                if len(entities) != len(emotions):
                    entities = [""]
                    emotions = [""]
                result[news_id] = (entities, emotions)
            except ValueError:
                continue

    return result


def clean_result(src_file, tgt_file):
    result = load_result(src_file)
    ner_dict = load_nerDict()

    with open(tgt_file, 'w') as tgt:
        for news_id, (entities, emotions) in result.items():
            new_entities = []
            new_emotions = []
            for entity, emotion in zip(entities, emotions):
                if entity in ner_dict:
                    new_entities.append(entity)
                    new_emotions.append(emotion)
            line = news_id + '\t' + \
                ','.join(new_entities) + '\t' + ','.join(new_emotions) + '\n'
            tgt.write(line)
    print("Done")


def truncate_result(src, tgt, length=2):
    src_result = load_result(src)
    with open(tgt, 'w') as f:
        for news_id, ee_pair in src_result.items():
            line = news_id + '\t' + ','.join(ee_pair[0][:length]) +
            '\t' + ','.join(ee_pair[1][:length]) + '\n'
            f.write(line)
    print("Done!")


def count_results(result_file):
    """统计实体长度"""
    counts = []
    results = load_result(result_file)
    for news_id, ee_pair in results.items():
        counts.append(len(ee_pair[0]))
    return Counter(counts)


def remove_repeated(src, tgt):
    """消除重复的实体"""
    src_result = load_result(src)
    with open(tgt, 'w') as f:
        for news_id, (entities, emotions) in src_result.items():
            clean_entities = []  # 存储非重复的实体
            clean_emotions = []
            for entity, emotion in zip(entities, emotions):
                if entity not in clean_entities:
                    clean_entities.append(entity)
                    clean_emotions.append(emotion)

            line = news_id + '\t' + ','.join(clean_entities) + \
                '\t' + ','.join(clean_emotions) + '\n'
            f.write(line)


def ensemble_results_counts(results, tgt_file):
    """通过出现的次数分数进行ensemble"""

    all_results = {}
    for r_file in results:
        result = load_result(r_file)
        for news_id, (entities, emotions) in result.items():
            if news_id not in all_results:
                all_results[news_id] = [entities, emotions]
            else:
                all_results[news_id][0] += entities
                all_results[news_id][1] += emotions

    # 需要article, all_articles, nerDict
    nerDict = load_nerDict()
    articles = load_test_articles()
    print("ensembling...")
    tgt = open(tgt_file, 'w')
    for news_id, (entities, emotions) in all_results.items():
        article = articles[news_id]
        entities, emotions = choose_core_entities(
            entities, emotions, article, nerDict
        )
        line = news_id + '\t' + ','.join(entities) + \
            '\t' + ','.join(emotions) + '\n'
        tgt.write(line)
    print("Done!")
    tgt.close()


def ensemble_crf_results(results, tgt_file):

    print("Ensembling...")
    all_results = {}
    for r_file in results:
        result = load_result(r_file)
        for news_id, (entities, emotions) in result.items():
            if news_id not in all_results:
                all_results[news_id] = [entities, emotions]
            else:
                all_results[news_id][0] += entities
                all_results[news_id][1] += emotions

    # 对重复的实体进行过滤
    for news_id, (entities, emotions) in all_results.items():
        all_pairs = [(entity, emotion)
                     for entity, emotion in zip(entities, emotions)]
        all_pairs = Counter(all_pairs)

        new_entities = []
        new_emotions = []
        for (entity, emotion), count in all_pairs.most_common():
            if entity not in new_entities and len(entity) > 1 and count > 1:
                new_entities.append(entity)
                new_emotions.append(emotion)
        all_results[news_id] = [new_entities, new_emotions]

    # 写入
    tgt = open(tgt_file, 'w')
    for news_id, (entities, emotions) in all_results.items():

        line = news_id + '\t' + ','.join(entities) + \
            '\t' + ','.join(emotions) + '\n'
        tgt.write(line)
    print("Done!")
    tgt.close()


def load_test_articles():
    articles = {}
    path = "SohuData/original_data/test_char.txt"
    with open(path) as f:
        for line in f:
            json_data = json.loads(line)
            id_ = json_data['newsId']
            article = json_data['title'] + "。" + json_data['content']
            article = clean(article)
            articles[id_] = article

    return articles


def ensemble_results_naive(src1, src2, tgt, allow_single=False):
    """
       ensemble lstm+crf 以及 lstm+lstm的结果
       ensemble的策略:
        情况1：crf为空的新闻，从lstm的结果中选择两个实体加入
        情况2：crf实体数为1的新闻，从lstm的结果中选择第一个实体查看是否重复,不重复则添加
    """

    result1 = load_result(src1)
    result2 = load_result(src2)

    with open(tgt, 'w') as f:
        for news_id, (entities, emotions) in result1.items():
            if len(entities) == 1:
                # print(entities)
                if entities[0] == "":  # 情况1
                    final_entities, final_emotions = result2[news_id]
                else:  # 情况2
                    final_entities = entities
                    final_emotions = emotions

                    if not allow_single:
                        entities2, emotions2 = result2[news_id]
                        for entity, emotion in zip(entities2, emotions2):
                            if entity not in final_entities:
                                final_entities.append(entity)
                                final_emotions.append(emotion)

            else:
                final_entities = entities
                final_emotions = emotions

            line = news_id + '\t' + ','.join(final_entities) + \
                '\t' + ','.join(final_emotions) + '\n'
            f.write(line)
    print("Done！")


def flatten_lists(lists):
    flatten_list = []
    for l in lists:
        if type(l) == list:
            flatten_list += l
        else:
            flatten_list.append(l)
    return flatten_list


if __name__ == "__main__":

    # 获取参数
    # parser = argparse.ArgumentParser(description="加载模型并解码")
    # parser.add_argument('--method', required=True, help='The name of model')
    # args = parser.parse_args()
    # method = args.method
    # assert method in ['lstm', 'lstm_crf', 'lstm_lstm']

    # test_json = "SohuData/original_data/coreEntityEmotion_test_stage1.txt"
    # model_path = "ckpts/{}.pkl".format(method)
    # save_file = "Results/{}/result.txt".format(method)

    # print("Loading model {} and decoding...".format(method))
    # load_and_decoding(model_path, test_json, save_file, method)
    pass
