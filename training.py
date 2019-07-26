from collections import Counter
from os.path import join
import json
import os

from config import TrainingConfig, LSTMConfig
from models.bilstm_crf import BILSTM_Model
from utils.model import save_model, get_meta
from utils.evaluation import Metrics
from utils.decoding import flatten_lists, decoding
from utils.analysis import analysis_result
from utils.util import convert_loader, back_map


def bilstm_train_and_eval(train_loader, dev_loader, eval_loader,
                          test_loader, token2id, tag2id, method):
    """训练并保存模型"""

    vocab_size = len(token2id)
    out_size = len(tag2id)
    meta = get_meta([TrainingConfig.__dict__, LSTMConfig.__dict__])

    model = BILSTM_Model(vocab_size, out_size, token2id, tag2id, method=method)
    model.train(train_loader, dev_loader, eval_loader)

    try:
        # 保存模型的信息
        root_dir = "/home/luopx/share_folders/Sohu"
        model_dir = 'ckpts/{}/{}-{}-Len{}-{:.2f}-{:.4f}'.format(
            model.method,
            meta['token_method'],
            meta['tag_schema'],
            meta['max_len'],
            model.best_val_loss,
            model.best_f1_score
        )
        model_dir = join(root_dir, model_dir)

        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        save_model(model, join(model_dir, "model.pkl"))

        # 保存word2id  tag2id 以及模型设置的信息
        with open(join(model_dir, 'meta.json'), 'w') as w:
            w.write(json.dumps(meta, indent=4))

        # 在验证集上面观察模型的效果、特点
        print("评估{}模型中...".format(method))
        # 分析结果
        print("分析在验证集上的结果...")
        metrics = model.cal_scores(eval_loader, use_model='best_f1')
        with open(join(model_dir, 'dev_result.txt'), 'w') as outfile:
            metrics.report_details(outfile=outfile)

        # 加载测试集，解码，将结果保存成文件
        print("在val_loss最小的模型上解码...")
        test_result = join(model_dir, 'min_devLoss_result.txt')
        decoding(model, test_loader, test_result)
        print("在f1分数值最大的模型上解码...")
        test_result = join(model_dir, 'max_f1_result.txt')
        decoding(model, test_loader, test_result, use_model="best_f1")

    except:
        import pdb
        pdb.set_trace()
