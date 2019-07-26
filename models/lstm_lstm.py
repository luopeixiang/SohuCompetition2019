import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

INIT = 1e-2

# 模型参考自 https://arxiv.org/abs/1706.05075


class DoubleLSTM(nn.Module):
    def __init__(self, vocab_size,
                 emb_size,
                 enc_hidden_size,
                 dec_hidden_size,
                 out_size,
                 dropout_p):
        super(DoubleLSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.dropout = nn.Dropout(p=dropout_p)
        self.encoder = nn.LSTM(emb_size, enc_hidden_size,
                               batch_first=True,
                               bidirectional=True)

        self.decoder = nn.LSTMCell(enc_hidden_size*2,
                                   dec_hidden_size)

        # 符号与原论文中的数学符号保持一致
        self.W_ts = nn.Linear(dec_hidden_size, out_size)
        self.W_y = nn.Linear(out_size, out_size)

        # 将encoder的初始状态 设置为可学习的参数
        self._init_enc_h = nn.Parameter(
            torch.Tensor(2, enc_hidden_size)  # 2表示双向
        )
        self._init_enc_c = nn.Parameter(
            torch.Tensor(2, enc_hidden_size)  # 2表示双向
        )
        nn.init.uniform_(self._init_enc_h, -INIT, INIT)
        nn.init.uniform_(self._init_enc_c, -INIT, INIT)

        # 将encoder最后时刻的输出h_n, c_n，
        # 映射成decoder的初始状态(dec_h_0, dec_c_0)
        self.projection = nn.Linear(enc_hidden_size*2, dec_hidden_size)

        # 将dec_h与tag_embedding拼接起来的结果投射成下一时刻的dec_h
        self.prejection_next_dec_h = nn.Linear(dec_hidden_size+out_size,
                                               dec_hidden_size)

    def forward(self, sents_tensor, lengths):

        enc_out, enc_final_states = self.encode(sents_tensor, lengths)

        # decoding
        outputs = self.decode(enc_out, enc_final_states)

        return outputs

    def encode(self, sents_tensor, lengths):
        emb = self.embedding(sents_tensor)
        emb = self.dropout(emb)

        # 构建初始化隐向量
        batch_size = emb.size(0)
        size = (
            self._init_enc_h.size(0),
            batch_size,
            self._init_enc_h.size(1)
        )
        init_enc_states = (
            self._init_enc_h.unsqueeze(1).expand(*size).contiguous(),
            self._init_enc_c.unsqueeze(1).expand(*size).contiguous()
        )

        # encoding
        packed = pack_padded_sequence(emb, lengths, batch_first=True)
        enc_out, (enc_hn, enc_cn) = self.encoder(packed, init_enc_states)
        enc_out, _ = pad_packed_sequence(
            enc_out, batch_first=True)
        # enc_out here: [B, L, hidden_size*2]
        # enc_hn and enc_cn: [2, B, hidden_size]

        # 将两个方向的状态拼接到一起
        final_states = (
            # [B, enc_hidden_size*2]
            torch.cat(enc_hn.chunk(2, dim=0), dim=2).squeeze(),
            torch.cat(enc_cn.chunk(2, dim=0), dim=2).squeeze()
        )

        return enc_out, final_states

    def decode(self, enc_out, enc_final_states):
        enc_final_h, enc_final_c = enc_final_states
        enc_out = enc_out.transpose(0, 1)  # [L, B, enc_hidden_size*2]

        # 构建初始化状态
        dec_h = self.projection(enc_final_h)  # [B, dec_hidden_size]
        dec_c = self.projection(enc_final_c)
        tag_embedding = torch.tanh(
            self.W_ts(dec_h))  # [B, out_size]

        dec_h = self.prejection_next_dec_h(
            torch.cat([dec_h, tag_embedding], dim=1)
        )  # [B, dec_hidden_size]

        # 使用LSTMCell，每一步将tag_embedding添加到输入
        max_len = enc_out.size(0)
        outputs = []
        for step in range(max_len):
            dec_input = enc_out[step]  # [B, enc_hidden_size*2]
            dec_h, dec_c = self.decoder(dec_input, (dec_h, dec_h))

            # 生成当前时刻的tag_embedding
            tag_embedding = torch.tanh(self.W_ts(dec_h))
            step_out = self.W_y(tag_embedding)  # [B, out_size]
            outputs.append(step_out)

            # 生成下一时刻的dec_h
            dec_h = self.prejection_next_dec_h(
                torch.cat([dec_h, tag_embedding], dim=1)
            )  # [B, dec_hidden_size]

        # 处理outputs
        outputs = torch.stack(outputs, dim=1)  # [B, max_len, out_size]

        return outputs

    def test(self, sents_tensor, lengths, _):
        logits = self.forward(sents_tensor, lengths)
        _, batch_tagids = torch.max(logits, dim=2)

        return batch_tagids
