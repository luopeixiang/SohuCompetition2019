from itertools import zip_longest
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from gensim.models import KeyedVectors

from utils.model import cal_lstm_crf_loss, \
    cal_weighted_loss, time_format, get_optimizer, change_embedding_lr
from config import TrainingConfig, LSTMConfig
from utils.evaluation import EntityEmotionMetrics
from utils.util import back_map, convert_loader
from .bilstm import BiLSTM
from .lstm_lstm import DoubleLSTM


class BILSTM_Model(object):
    def __init__(self, vocab_size, out_size, token2id, tag2id, method="lstm"):
        """功能：对LSTM的模型进行训练与测试
           参数:
            vocab_size:词典大小
            out_size:标注种类
            method: 三种方法，["lstm", "lstm_crf", "lstm_lstm"]
            crf选择是否添加CRF层"""
        self.method = method
        self.out_size = out_size
        self.token2id = token2id
        self.tag2id = tag2id
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        assert method in ["lstm", "lstm_crf", "lstm_lstm"]

        # 加载模型参数
        self.emb_size = LSTMConfig.emb_size
        self.max_len = TrainingConfig.max_len
        self.clip = TrainingConfig.clip

        # 根据是否添加crf初始化不同的模型 选择不一样的损失计算函数
        if method == "lstm":
            self.hidden_size = LSTMConfig.hidden_size
            self.model = BiLSTM(vocab_size, self.emb_size,
                                self.hidden_size, out_size).to(self.device)
            self.alpha = TrainingConfig.alpha
            self.cal_loss_func = cal_weighted_loss
        elif method == "lstm_crf":
            self.hidden_size = LSTMConfig.hidden_size
            self.model = BiLSTM_CRF(vocab_size, self.emb_size,
                                    self.hidden_size, out_size).to(self.device)
            self.cal_loss_func = cal_lstm_crf_loss
        elif method == "lstm_lstm":
            self.enc_hsize = LSTMConfig.enc_hidden_size
            self.dec_hsize = LSTMConfig.dec_hidden_size
            self.dropout_p = LSTMConfig.dropout_p
            self.model = DoubleLSTM(vocab_size,
                                    self.emb_size,
                                    self.enc_hsize,
                                    self.dec_hsize,
                                    out_size,
                                    self.dropout_p
                                    )
            self.alpha = TrainingConfig.alpha
            self.cal_loss_func = cal_weighted_loss

        self.model = self.model.to(self.device)

        # 加载训练参数：
        self.patience = TrainingConfig.training_patience
        self.print_step = TrainingConfig.print_step
        self.lr = TrainingConfig.lr
        self.lr_patience = TrainingConfig.lr_patience
        self.lr_decay = TrainingConfig.lr_decay
        self.batch_size = TrainingConfig.batch_size
        self.decoding_batch_size = TrainingConfig.decoding_batch_size

        self.embedding_lr = TrainingConfig.embedding_lr
        self.embedding_le = TrainingConfig.embedding_learning_epoch
        self.token_method = TrainingConfig.token_method

        # 初始化优化器 一开始embedding层学习速率设为0
        self.optimizer = get_optimizer(self.model, 0.0, self.lr)
        # 初始化其他指标
        self.step = 0
        self.best_val_loss = 1e18
        self.best_model = None

        # 最好的分数(f1分数)
        self.best_metrics = None
        self.best_f1_model = None
        self.best_f1_score = 0.

    def train(self, train_loader, dev_loader, eval_loader):
        self._init_embeddings(self.token2id)
        # 对数据集按照长度进行排序

        epoch = 1  # epoch
        patience_count = 0
        start = time.time()
        while True:
            self.step = 0
            losses = 0.
            if epoch == self.embedding_le:
                print("Start Learning Embedding!")
                self.optimizer = change_embedding_lr(
                    self.optimizer, self.embedding_lr)

            for articles, tags, lengths in train_loader:
                losses += self.train_step(articles, tags, lengths)
                if self.step % TrainingConfig.print_step == 0:
                    total_step = len(train_loader)
                    print("Epoch {}, step/total_step: {}/{} {:.2f}% Loss:{:.4f}".format(
                        epoch, self.step, total_step,
                        100. * self.step / total_step,
                        losses / self.print_step
                    ))
                    losses = 0.

            # 每轮结束测试在验证集上的性能，保存最好的一个
            val_loss = self.validate(dev_loader)
            print("Epoch {}, Val Loss:{:.4f}".format(epoch, val_loss))

            # 计算验证集上的f1分数，保存最好的一个
            metrics = self.cal_scores(eval_loader)
            metrics.report()
            if metrics.final_f1 > self.best_f1_score:
                print("更新f1并保存模型...")
                self.best_metrics = metrics
                self.best_f1_model = copy.deepcopy(self.model)
                assert id(self.best_f1_model) != id(self.model)  # 确认有复制
                self.best_f1_score = metrics.final_f1
                patience_count = 0  # 有改进，patience_count清零
            else:
                patience_count += 1

                # 如果连续lr_patience个回合f1分数下降,将当前模型换成best_f1_model
                # 并且重新初始化学习器 ,降低学习率为原来的一半
                if patience_count >= self.lr_patience:
                    print("Reduce Learning Rate....")
                    self.embedding_lr *= self.lr_decay
                    self.lr *= self.lr_decay
                    self.optimizer = get_optimizer(
                        self.model, self.embedding_lr, self.lr)
                    self.model = copy.deepcopy(self.best_f1_model)

                # 如果连续patience个回合分数下降，则结束
                if patience_count >= self.patience:
                    end = time.time()
                    print("在Epoch {} 训练终止,用时: {}".format(
                        epoch, time_format(end-start)))
                    break  # 终止训练

            epoch += 1

    def train_step(self, articles, tags, lengths):

        self.model.train()
        self.step += 1
        # 准备数据
        articles = articles.to(self.device)
        tags = tags.to(self.device)

        # forward
        scores = self.model(articles, lengths)

        # 计算损失 更新参数
        self.optimizer.zero_grad()
        loss = self.cal_loss_func(scores, tags, self.tag2id).to(self.device)
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

        return loss.item()

    def validate(self, dev_loader):
        self.model.eval()
        with torch.no_grad():
            val_losses = 0.
            val_step = 0
            for articles, tags, lengths in dev_loader:
                val_step += 1
                # 准备batch数据
                articles = articles.to(self.device)
                tags = tags.to(self.device)

                # forward
                scores = self.model(articles, lengths)

                # 计算损失
                loss = self.cal_loss_func(
                    scores, tags, self.tag2id).to(self.device)
                val_losses += loss.item()
            val_loss = val_losses / val_step

            if val_loss < self.best_val_loss:
                print("保存模型...")
                self.best_model = copy.deepcopy(self.model)
                self.best_val_loss = val_loss

            return val_loss

    def cal_scores(self, data_loader, use_model='best_val_loss'):

        # 从dev_loader构建article_loader
        pred_tags, articles_ids, articles_truncated = self.test(
            data_loader, use_model=use_model)
        metrics = EntityEmotionMetrics(
            pred_tags, articles_truncated, articles_ids, self.token_method)
        return metrics

    def test(self, article_loader, use_model='best_val_loss'):
        """返回最佳模型在测试集上的预测结果"""
        assert use_model in ['best_val_loss', 'best_f1']
        if use_model == 'best_val_loss':
            decoding_model = self.best_model
        else:
            decoding_model = self.best_f1_model

        all_pred_tags = []  # 存储结果，文本形式
        all_art_ids = []
        all_articles = []
        decoding_model.eval()
        with torch.no_grad():
            drop_last = True if self.method == "lstm_crf" else False
            for articles, article_ids, lengths in article_loader:
                articles = articles.to(self.device)
                pred_tags = decoding_model.test(
                    articles, lengths, self.tag2id)
                all_pred_tags += back_map(pred_tags, lengths,
                                          self.tag2id, drop_last=drop_last)
                all_articles += back_map(articles, lengths,
                                         self.token2id, drop_last=drop_last)
                all_art_ids += article_ids

        return all_pred_tags, all_art_ids, all_articles

    def _init_embeddings(self, word2id):

        print('Loading Pretrained embedding...')
        wv = KeyedVectors.load_word2vec_format(TrainingConfig.word2vec_path)
        if self.method == "lstm_crf":
            for word, id_ in word2id.items():
                try:
                    self.model.bilstm.embedding.weight.data[id_] = torch.Tensor(
                        wv[word])
                except KeyError:
                    continue
        else:
            for word, id_ in word2id.items():
                try:
                    self.model.embedding.weight.data[id_] = torch.Tensor(
                        wv[word])
                except KeyError:
                    continue


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, out_size):
        """初始化参数：
            vocab_size:字典的大小
            emb_size:词向量的维数
            hidden_size：隐向量的维数
            out_size:标注的种类
        """
        super(BiLSTM_CRF, self).__init__()
        self.bilstm = BiLSTM(vocab_size, emb_size, hidden_size, out_size)

        # CRF实际上就是多学习一个转移矩阵 [out_size, out_size] 初始化为均匀分布
        self.transition = nn.Parameter(
            torch.ones(out_size, out_size) * 1/out_size)
        # self.transition.data.zero_()

    def forward(self, sents_tensor, lengths):
        # [B, L, out_size]
        emission = self.bilstm(sents_tensor, lengths)

        # 计算CRF scores, 这个scores大小为[B, L, out_size, out_size]
        # 也就是每个字对应对应一个 [out_size, out_size]的矩阵
        # 这个矩阵第i行第j列的元素的含义是：上一时刻tag为i，这一时刻tag为j的分数
        batch_size, max_len, out_size = emission.size()
        crf_scores = emission.unsqueeze(
            2).expand(-1, -1, out_size, -1) + self.transition.unsqueeze(0)

        return crf_scores

    def test(self, test_sents_tensor, lengths, tag2id):
        """使用维特比算法进行解码"""
        start_id = tag2id['<start>']
        end_id = tag2id['<end>']
        pad = tag2id['<pad>']
        tagset_size = len(tag2id)

        crf_scores = self.forward(test_sents_tensor, lengths)
        device = crf_scores.device
        # B:batch_size, L:max_len, T:target set size
        B, L, T, _ = crf_scores.size()
        # viterbi[i, j, k]表示第i个句子，第j个字对应第k个标记的最大分数
        viterbi = torch.zeros(B, L, T).to(device)
        # backpointer[i, j, k]表示第i个句子，第j个字对应第k个标记时前一个标记的id，用于回溯
        backpointer = (torch.zeros(B, L, T).long() * end_id).to(device)
        lengths = torch.LongTensor(lengths).to(device)
        # 向前递推
        for step in range(L):
            batch_size_t = (lengths > step).sum().item()
            if step == 0:
                # 第一个字它的前一个标记只能是start_id
                viterbi[:batch_size_t, step,
                        :] = crf_scores[: batch_size_t, step, start_id, :]
                backpointer[: batch_size_t, step, :] = start_id
            else:
                max_scores, prev_tags = torch.max(
                    viterbi[:batch_size_t, step-1, :].unsqueeze(2) +
                    crf_scores[:batch_size_t, step, :, :],     # [B, T, T]
                    dim=1
                )
                viterbi[:batch_size_t, step, :] = max_scores
                backpointer[:batch_size_t, step, :] = prev_tags

        # 在回溯的时候我们只需要用到backpointer矩阵
        backpointer = backpointer.view(B, -1)  # [B, L * T]
        tagids = []  # 存放结果
        tags_t = None
        for step in range(L-1, 0, -1):
            batch_size_t = (lengths > step).sum().item()
            if step == L-1:
                index = torch.ones(batch_size_t).long() * (step * tagset_size)
                index = index.to(device)
                index += end_id
            else:
                prev_batch_size_t = len(tags_t)

                new_in_batch = torch.LongTensor(
                    [end_id] * (batch_size_t - prev_batch_size_t)).to(device)
                offset = torch.cat(
                    [tags_t, new_in_batch],
                    dim=0
                )  # 这个offset实际上就是前一时刻的
                index = torch.ones(batch_size_t).long() * (step * tagset_size)
                index = index.to(device)
                index += offset.long()

            try:
                tags_t = backpointer[:batch_size_t].gather(
                    dim=1, index=index.unsqueeze(1).long())
            except RuntimeError:
                import pdb
                pdb.set_trace()
            tags_t = tags_t.squeeze(1)
            tagids.append(tags_t.tolist())

        # tagids:[L-1]（L-1是因为扣去了end_token),大小的liebiao
        # 其中列表内的元素是该batch在该时刻的标记
        # 下面修正其顺序，并将维度转换为 [B, L]
        tagids = list(zip_longest(*reversed(tagids), fillvalue=pad))
        tagids = torch.Tensor(tagids).long()

        # 返回解码的结果
        return tagids
