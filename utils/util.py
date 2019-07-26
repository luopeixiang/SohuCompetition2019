from os.path import join
import json
import pickle
import difflib


def load_maps(path, method, tag_schema):
    with open(join(path, '{}2id.json'.format(method))) as f:
        word2id = json.loads(f.read())

    with open(join(path, '{}_tag2id.json'.format(tag_schema))) as f:
        tag2id = json.loads(f.read())

    return word2id, tag2id


def save_model(model, file_name):
    """用于保存模型"""
    with open(file_name, "wb") as f:
        pickle.dump(model, f)


def load_model(file_name):
    """用于加载模型"""
    with open(file_name, "rb") as f:
        model = pickle.load(f)
    return model


def flatten_lists(lists):
    flatten_list = []
    for l in lists:
        if type(l) == list:
            flatten_list += l
        else:
            flatten_list.append(l)
    return flatten_list


def get_meta(configs):
    """获取模型信息"""
    meta = {}
    for config_dict in configs:
        for key, value in config_dict.items():
            if not key.startswith("__"):
                meta[key] = value
    return meta


def back_map(ids_, lengths, map_, drop_last=False):
    """反映射，将数字id表示的tag或者word转化为源文本"""
    if drop_last:
        lengths = [length-1 for length in lengths]

    bmap = dict((v, k) for k, v in map_.items())
    if type(ids_) != list:
        ids_ = ids_.tolist()

    results = []
    for id_list, length in zip(ids_, lengths):
        result = [bmap[id_] for id_ in id_list][:length]
        results.append(result)

    return results


def convert_loader(labeled_loader):
    """labeled_loader --> unlabeled_loader for model.test function"""

    unlabeled_loader = []
    all_golden_tags = []
    none_id = [None]
    all_lengths = []
    for batch_articles, batch_tags, batch_lengths in labeled_loader:
        unlabeled_loader.append(
            (batch_articles, none_id * len(batch_lengths), batch_lengths))
        all_golden_tags += batch_tags.tolist()
        all_lengths += batch_lengths

    return unlabeled_loader, all_golden_tags, all_lengths


def get_edit_distance(str1, str2):
    """衡量两个字符串之间的编辑距离
    参考：https://www.jianshu.com/p/466cf6624e26
    """
    leven_cost = 0
    s = difflib.SequenceMatcher(None, str1, str2)
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == 'replace':
            leven_cost += max(i2-i1, j2-j1)
        elif tag == 'insert':
            leven_cost += (j2-j1)
        elif tag == 'delete':
            leven_cost += (i2-i1)
    return leven_cost
