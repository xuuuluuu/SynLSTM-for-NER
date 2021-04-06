import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple


def masked_flip(padded_sequence: torch.Tensor, sequence_lengths: List[int]) -> torch.Tensor:
    """
        Flips a padded tensor along the time dimension without affecting masked entries.
        # Parameters
        padded_sequence : `torch.Tensor`
            The tensor to flip along the time dimension.
            Assumed to be of dimensions (batch size, num timesteps, ...)
        sequence_lengths : `torch.Tensor`
            A list containing the lengths of each unpadded sequence in the batch.
        # Returns
        A `torch.Tensor` of the same shape as padded_sequence.
        """
    assert padded_sequence.size(0) == len(
        sequence_lengths
    ), f"sequence_lengths length ${len(sequence_lengths)} does not match batch size ${padded_sequence.size(0)}"
    num_timesteps = padded_sequence.size(1)
    flipped_padded_sequence = torch.flip(padded_sequence, [1])
    sequences = [
        flipped_padded_sequence[i, num_timesteps - length :]
        for i, length in enumerate(sequence_lengths)
    ]
    return torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)


class DepLabeledGCN(nn.Module):
    def __init__(self, config, hidden_dim, input_dim, graph_dim):
        super().__init__()
        self.lstm_hidden = hidden_dim//2
        self.input_dim = input_dim
        self.graph_dim = graph_dim
        self.device = config.device
        self.gcn_layer = config.num_gcn_layers
        self.drop_lstm = nn.Dropout(config.dropout).to(self.device)
        self.drop_gcn = nn.Dropout(config.dropout).to(self.device)

        self.lstm_f = MyLSTM(self.input_dim, self.lstm_hidden, self.graph_dim).to(self.device)
        self.lstm_b = MyLSTM(self.input_dim, self.lstm_hidden, self.graph_dim).to(self.device)
        # self.lstm1 = MyLSTM(200, 100).to(self.device)
        # self.lstm_b1 = MyLSTM(200, 100).to(self.device)
        self.W = nn.ModuleList()
        for layer in range(self.gcn_layer):
            self.W.append(nn.Linear(self.graph_dim, self.graph_dim)).to(self.device)

    def forward(self, inputs, word_seq_len, adj_matrix, dep_label_matrix):

        """

        :param gcn_inputs:
        :param word_seq_len:
        :param adj_matrix: should already contain the self loop
        :param dep_label_matrix:
        :return:
        """
        adj_matrix = adj_matrix.to(self.device)
        batch_size, sent_len, input_dim = inputs.size()
        denom = adj_matrix.sum(2).unsqueeze(2) + 1

        graph_input = inputs[:, :, :self.graph_dim]

        for l in range(self.gcn_layer):
            Ax = adj_matrix.bmm(graph_input)  ## N x N  times N x h  = Nxh
            AxW = self.W[l](Ax)   ## N x m
            AxW = AxW + self.W[l](graph_input)  ## self loop  N x h
            AxW = AxW / denom
            graph_input = torch.relu(AxW)
        
        # forward LSTM
        lstm_out = self.lstm_f(inputs, graph_input)
        # backward LSTM
        word_rep_b = masked_flip(inputs, word_seq_len.tolist())
        c_b = masked_flip(graph_input, word_seq_len.tolist())
        lstm_out_b = self.lstm_b(word_rep_b, c_b)
        lstm_out_b = masked_flip(lstm_out_b, word_seq_len.tolist())

        feature_out = torch.cat((lstm_out, lstm_out_b), dim=2)
        feature_out = self.drop_lstm(feature_out)

        return feature_out
        
class MyLSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz, g_sz):
        super(MyLSTM, self).__init__()
        self.input_sz = input_sz
        self.hidden_sz = hidden_sz
        self.g_sz = g_sz
        self.all1 = nn.Linear((self.hidden_sz * 1 + self.input_sz  * 1),  self.hidden_sz)
        self.all2 = nn.Linear((self.hidden_sz * 1 + self.input_sz  +self.g_sz), self.hidden_sz)
        self.all3 = nn.Linear((self.hidden_sz * 1 + self.input_sz  +self.g_sz), self.hidden_sz)
        self.all4 = nn.Linear((self.hidden_sz * 1 + self.input_sz  * 1), self.hidden_sz)

        self.all11 = nn.Linear((self.hidden_sz * 1 + self.g_sz),  self.hidden_sz)
        self.all44 = nn.Linear((self.hidden_sz * 1 + self.g_sz), self.hidden_sz)

        self.init_weights()
        self.drop = nn.Dropout(0.5)
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_sz)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def node_forward(self, xt, ht, Ct_x, mt, Ct_m):

        # # # new standard lstm
        hx_concat = torch.cat((ht, xt), dim=1)
        hm_concat = torch.cat((ht, mt), dim=1)
        hxm_concat = torch.cat((ht, xt, mt), dim=1)


        i = self.all1(hx_concat)
        o = self.all2(hxm_concat)
        f = self.all3(hxm_concat)
        u = self.all4(hx_concat)
        ii = self.all11(hm_concat)
        uu = self.all44(hm_concat)

        i, f, o, u = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(u)
        ii,uu = torch.sigmoid(ii), torch.tanh(uu)
        Ct_x = i * u + ii * uu + f * Ct_x
        ht = o * torch.tanh(Ct_x) 

        return ht, Ct_x, Ct_m 

    def forward(self, x, m, init_stat=None):
        batch_sz, seq_sz, _ = x.size()
        hidden_seq = []
        cell_seq = []
        if init_stat is None:
            ht = torch.zeros((batch_sz, self.hidden_sz)).to(x.device)
            Ct_x = torch.zeros((batch_sz, self.hidden_sz)).to(x.device)
            Ct_m = torch.zeros((batch_sz, self.hidden_sz)).to(x.device)
        else:
            ht, Ct = init_stat
        for t in range(seq_sz):  # iterate over the time steps
            xt = x[:, t, :]
            mt = m[:, t, :]
            ht, Ct_x, Ct_m= self.node_forward(xt, ht, Ct_x, mt, Ct_m)
            hidden_seq.append(ht)
            cell_seq.append(Ct_x)
            if t == 0:
                mht = ht
                mct = Ct_x
            else:
                mht = torch.max(torch.stack(hidden_seq), dim=0)[0]
                mct = torch.max(torch.stack(cell_seq), dim=0)[0]
        hidden_seq = torch.stack(hidden_seq).permute(1, 0, 2) ##batch_size x max_len x hidden
        return hidden_seq



