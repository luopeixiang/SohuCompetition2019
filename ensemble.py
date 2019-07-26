
# 对新家的东西进行测试
from functools import partial
from os.path import join
import json
import os

from torch.utils.data import DataLoader

from utils.model import get_meta, load_model
from config import TrainingConfig, LSTMConfig
from data.dataset import SohuDataSet, collate_tagged
from utils.analysis import *
from utils.util import convert_loader, back_map
from data.dataset import SohuTestDataSet, collate_untagged
from utils.decoding import *


def ensemble(ckpts_path, tgt, min_score=0.0, max_len=1000):
    """疯狂ensemble
    ckpts_path:设置模型路径
    tgt:设置结果保存的路径
    min_score和max_len作为ensemble的模型的条件
    """
    result_files = []
    for ckpt in os.listdir(ckpts_path):
        score = float(ckpt.split('-')[-1])
        length = int(ckpt.split('-')[-3][3:])
        if score > min_score and length < max_len:
            result_files.append(join(ckpts_path, ckpt, "max_f1_result.txt"))

    print("共ensemble {} 个文件:".format(len(result_files)))
    for f in result_files:
        print(f)
    ensemble_results_counts(result_files, tgt)


if __name__ == "__main__":

    ensemble("ckpts/lstm_crf",
             "results/ensemble.txt",
             min_score=0.38,  # 选择在验证集上分数大于0.38的模型
             max_len=280
             )

    root_dir = "./results"
    result1 = join(root_dir, "ensemble.txt")
    result2 = join(root_dir, "best_now.txt")
    ensemble_results_naive(
        result1, result2, join(root_dir, "tmp.txt"), allow_single=False)
