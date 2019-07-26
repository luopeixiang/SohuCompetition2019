import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, out_size):
        """初始化参数：
            vocab_size:字典的大小
            emb_size:词向量的维数
            hidden_size：隐向量的维数
            out_size:标注的种类
        """
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.bilstm = nn.LSTM(emb_size, hidden_size,
                              batch_first=True,
                              num_layers=1,         # 这两个是后续添加的
                              dropout=0.5,
                              bidirectional=True)

        # self.dropout = nn.Dropout(p=0.5)
        self.lin = nn.Linear(2*hidden_size, out_size)

    def forward(self, sents_tensor, lengths, return_rnn_out=False):
        emb = self.embedding(sents_tensor)  # [B, L, emb_size]
        # emb = self.dropout(emb)

        packed = pack_padded_sequence(emb, lengths, batch_first=True)
        rnn_out, _ = self.bilstm(packed)
        # rnn_out:[B, L, hidden_size*2]
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)

        # 试一下增加dropout
        # rnn_out = self.dropout(rnn_out)

        # 试一下增加非线性单元
        # rnn_out = torch.tanh(rnn_out)

        if return_rnn_out:
            return rnn_out

        scores = self.lin(rnn_out)  # [B, L, out_size]

        return scores

    def test(self, sents_tensor, lengths, _):
        """第三个参数不会用到，加它是为了与BiLSTM_CRF保持同样的接口"""
        logits = self.forward(sents_tensor, lengths)  # [B, L, out_size]
        _, batch_tagids = torch.max(logits, dim=2)

        return batch_tagids
