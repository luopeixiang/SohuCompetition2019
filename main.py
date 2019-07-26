from functools import partial

from torch.utils.data import DataLoader

from training import bilstm_train_and_eval
from config import TrainingConfig
from data.dataset import SohuDataSet, collate_tagged
from data.dataset import SohuTestDataSet, collate_untagged
from utils.util import load_maps


def main():
    """训练模型，评估结果"""

    method = TrainingConfig.method
    token_method = TrainingConfig.token_method
    tag_schema = TrainingConfig.tag_schema
    max_len = TrainingConfig.max_len

    print("训练模型：{}---{}---{}".format(method, token_method, tag_schema))

    # 读取数据
    print("读取数据...")
    token2id, tag2id = load_maps(
        "./SohuData/original_data/",
        token_method,
        tag_schema
    )

    train_loader = DataLoader(
        SohuDataSet(
            "train", "./SohuData/{}_{}/".format(token_method, tag_schema)),
        batch_size=TrainingConfig.batch_size,
        collate_fn=partial(collate_tagged, token2id, tag2id, method, max_len)
    )
    dev_loader = DataLoader(
        SohuDataSet(
            "dev", "./SohuData/{}_{}/".format(token_method, tag_schema)),
        batch_size=TrainingConfig.batch_size,
        collate_fn=partial(collate_tagged, token2id, tag2id, method, max_len)
    )
    eval_loader = DataLoader(
        SohuTestDataSet('SohuData/original_data', 'dev', token_method),
        batch_size=TrainingConfig.decoding_batch_size,
        collate_fn=partial(collate_untagged, token2id, method, max_len)
    )

    test_loader = DataLoader(
        SohuTestDataSet('SohuData/original_data', 'test', token_method),
        batch_size=TrainingConfig.decoding_batch_size,
        collate_fn=partial(collate_untagged, token2id, method, max_len)
    )

    print("正在训练评估{}模型...".format(method))
    bilstm_train_and_eval(
        train_loader,
        dev_loader,
        eval_loader,
        test_loader,
        token2id,
        tag2id,
        method
    )


if __name__ == "__main__":
    main()
