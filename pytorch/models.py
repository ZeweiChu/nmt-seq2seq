import torch
import torch.nn as nn
from torch.nn import Parameter 
import torch.nn.functional as F
from torch.autograd import Variable
import code
import math
import sys
import numpy as np
from utils import LinearND

class EncoderDecoderModel(nn.Module):
    def __init__(self, args):
        super(EncoderDecoderModel, self).__init__()
        self.nhid = args.hidden_size

        self.embed_en = nn.Embedding(args.en_total_words, args.embedding_size)
        self.embed_cn = nn.Embedding(args.cn_total_words, args.embedding_size)

        self.encoder_cell = nn.LSTMCell(args.embedding_size, args.hidden_size)
        self.decoder = nn.LSTM(args.embedding_size, args.hidden_size, batch_first=True)

        self.linear = LinearND(self.nhid, args.cn_total_words)
        self.linear.fc.bias.data.fill_(0)
        self.linear.fc.weight.data.uniform_(-0.1, 0.1)

        self.embed_en.weight.data.uniform_(-0.1, 0.1)
        self.embed_cn.weight.data.uniform_(-0.1, 0.1)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (weight.new(bsz, self.nhid).zero_(),
                weight.new(bsz, self.nhid).zero_())

    def forward(self, x, x_mask, y, hidden):
        
        x_embedded = self.embed_en(x)
        B, T, embedding_size = x_embedded.size()
        # encoder
        hiddens = []
        cells = []
        for i in range(T):
            hidden = self.encoder_cell(x_embedded[:,i,:], hidden)
            hiddens.append(hidden[0].unsqueeze(1))
            cells.append(hidden[1].unsqueeze(1))
        
        hiddens = torch.cat(hiddens, 1)
        cells = torch.cat(cells, 1)
        x_lengths = x_mask.sum(1).view(B, 1, 1).expand(B, 1, embedding_size)-1
        h = hiddens.gather(1, x_lengths).permute(1,0,2)
        c = cells.gather(1, x_lengths).permute(1,0,2)

        # decoder
        y_embedded = self.embed_cn(y)
        # code.interact(local=locals())
        hiddens, (h, c) = self.decoder(y_embedded, hx=(h, c))
        hiddens = hiddens.contiguous()
        # output layer
        decoded = self.linear(hiddens).view(B*hiddens.size(1), -1)
        decoded = F.log_softmax(decoded, -1)
        return decoded.view(hiddens.size(0), hiddens.size(1), decoded.size(1)), hiddens

    def translate(self, x, x_mask, y, hidden, max_length = 20):
        x_embedded = self.embed_en(x)
        B, T, embedding_size = x_embedded.size()
        # encoder
        hiddens = []
        cells = []
        for i in range(T):
            hidden = self.encoder_cell(x_embedded[:,i,:], hidden)
            hiddens.append(hidden[0].unsqueeze(1))
            cells.append(hidden[1].unsqueeze(1))
        
        hiddens = torch.cat(hiddens, 1)
        cells = torch.cat(cells, 1)
        x_lengths = x_mask.sum(1).view(B, 1, 1).expand(B, 1, embedding_size)-1
        h = hiddens.gather(1, x_lengths).permute(1,0,2)
        c = cells.gather(1, x_lengths).permute(1,0,2)

        pred = [y]
        for i in range(max_length-1):
            # print(i)
            # print(y.shape)
            y_embedded = self.embed_cn(y)
            # print(y_embedded.shape)
            # print(h.shape)
            # print(c.shape)
            hiddens, (h, c) = self.decoder(y_embedded, hx=(h, c))
            hiddens = hiddens.contiguous()
            # output layer
            decoded = self.linear(hiddens).view(B*hiddens.size(1), -1)
            decoded = F.log_softmax(decoded, -1)
            decoded = decoded.view(hiddens.size(0), decoded.size(1))
            y = torch.max(decoded, 1)[1].view(B, 1)
            pred.append(y)
        return torch.cat(pred, 1)

