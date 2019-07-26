from os.path import join

# 设置lstm训练参数


class TrainingConfig(object):
    # method = 'lstm'
    # method = 'lstm_crf'
    method = 'lstm_crf'

    batch_size = 64
    decoding_batch_size = 256  # 加快decode速度

    # 学习速率
    lr = 0.001
    print_step = 10
    # 调整学习率
    lr_decay = 0.5
    lr_patience = 2
    # The max gradient norm
    clip = 5.0

    # for lstm model and lstm_lstm model
    alpha = 5  # 用于设置实体标注的影响

    word2vec_path = '/home/luopx/DATA/sgns.sogou.char'

    # 若连续n个epoch f1分数没有提高，那么终止训练
    # training_patience指明n的大小
    training_patience = 4

    # 新设置！
    max_len = 300  # 发现长度越短越好...
    token_method = 'char'  # or 'word'
    # token_method = 'word'
    tag_schema = 'singleO'  # or 'multiO'

    # 学习率
    embedding_lr = 2e-4
    embedding_learning_epoch = 3  # 从第3个epoch才开始学习


class LSTMConfig(object):
    emb_size = 300  # 词向量的维数
    hidden_size = 300  # lstm隐向量的维数

    # for DoubleLSTM model
    enc_hidden_size = 300
    dec_hidden_size = 600
    dropout_p = 0.2
