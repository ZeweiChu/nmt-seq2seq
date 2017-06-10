import torch
import torch.nn as nn
from torch.nn import Parameter 
import torch.nn.functional as F
from torch.autograd import Variable
import code
import math
import sys
import numpy as np

class EncoderDecoderModel(nn.Module):
    def __init__(self, args):
        super(EncoderDecoderModel, self).__init__()
        self.nhid = args.hidden_size
        self.nlayers = args.num_layers

        self.embed_en = nn.Embedding(args.en_total_words, args.embedding_size)
        self.embed_cn = nn.Embedding(args.cn_total_words, args.embedding_size)
        self.encoder = nn.LSTM(args.embedding_size, args.hidden_size, batch_first=True)
        self.decoder = nn.LSTM(args.embedding_size, args.hidden_size, batch_first=True)

        self.linear = nn.Linear(self.nhid, args.cn_total_words)
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-0.1, 0.1)

        self.embed_en.weight.data.uniform_(-0.1, 0.1)
        self.embed_cn.weight.data.uniform_(-0.1, 0.1)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))

    def forward(self, x, x_mask, y, hidden):
        x_embedded = self.embed_en(x)
        # encoder
        hiddens, (h, c) = self.encoder(x_embedded, hidden)
        y_embedded = self.embed_cn(y)

        # decoder
        hiddens, (h, c) = self.decoder(y_embedded, hx=(h, c))

        hiddens = hiddens.contiguous()
        # output layer
        decoded = self.linear(hiddens.view(hiddens.size(0)*hiddens.size(1), hiddens.size(2)))
        decoded = F.log_softmax(decoded)
        return decoded.view(hiddens.size(0), hiddens.size(1), decoded.size(1)), hiddens

    def translate(self, x, x_mask, y, hidden, max_length = 20):
        x_embedded = self.embed_en(x)
        # encoder
        hiddens, (h, c) = self.encoder(x_embedded, hidden)
        
        pred = [y]
        for i in range(max_length-1):
            # code.interact(local=locals())
            y_embedded = self.embed_cn(y)
            hiddens, (h, c) = self.decoder(y_embedded, hx=(h, c))
            hiddens = hiddens.contiguous()
            # output layer
            decoded = self.linear(hiddens.view(hiddens.size(0)*hiddens.size(1), hiddens.size(2)))
            decoded = F.log_softmax(decoded)
            decoded = decoded.view(hiddens.size(0), decoded.size(1))
            y = torch.max(decoded, 1)[1]
            pred.append(y)
        return torch.cat(pred, 1)

