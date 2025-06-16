
import os, sys
import numpy as np
import paddle
# paddle.enable_static()
import paddle.fluid as fluid
from paddle import ParamAttr
# import paddle.nn as nn
# import paddle.nn.functional as F
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 666
input_size = 192
hidden_size = 96
num_embeddings = 38

class PPAttentionGRUCell(paddle.nn.Layer):
    def __init__(self, input_size, hidden_size, num_embeddings, use_gru=False):
        super(PPAttentionGRUCell, self).__init__()
        self.i2h = paddle.nn.Linear(input_size, hidden_size, bias_attr=False)
        self.h2h = paddle.nn.Linear(hidden_size, hidden_size)
        self.score = paddle.nn.Linear(hidden_size, 1, bias_attr=False)

        self.rnn = paddle.nn.GRUCell(
            input_size=input_size + num_embeddings, hidden_size=hidden_size)

        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):

        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = paddle.unsqueeze(self.h2h(prev_hidden), axis=1)

        res = paddle.add(batch_H_proj, prev_hidden_proj)
        res = paddle.tanh(res)
        e = self.score(res)

        alpha = paddle.nn.functional.softmax(e, axis=1)
        alpha = paddle.transpose(alpha, [0, 2, 1])
        context = paddle.squeeze(paddle.mm(alpha, batch_H), axis=1)
        concat_context = paddle.concat([context, char_onehots], 1)

        cur_hidden = self.rnn(concat_context, prev_hidden)

        return cur_hidden, alpha


class PTAttentionGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_embeddings, use_gru=False):
        super(PTAttentionGRUCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

        self.rnn = nn.GRUCell(
            input_size=input_size + num_embeddings, hidden_size=hidden_size, bias=True)

        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = torch.unsqueeze(self.h2h(prev_hidden), dim=1)

        res = torch.add(batch_H_proj, prev_hidden_proj)
        res = torch.tanh(res)
        e = self.score(res)

        alpha = F.softmax(e, dim=1)
        alpha = alpha.permute(0, 2, 1)
        context = torch.squeeze(torch.matmul(alpha, batch_H), dim=1)
        concat_context = torch.cat([context, char_onehots.float()], 1)

        cur_hidden = self.rnn(concat_context, prev_hidden)

        return (cur_hidden, cur_hidden), alpha


def paddle_grucell():
    np.random.seed(SEED)
    x = np.random.rand(1, 96).astype(np.float32)
    y = np.random.rand(1, 25, 192).astype(np.float32)
    z = np.zeros((1, 38)).astype(np.float32)
    z[0, 0] = 1

    # np.save('org.npy', x)
    with fluid.dygraph.guard():
        layer = PPAttentionGRUCell(input_size=input_size,
                                   hidden_size=hidden_size,
                                   num_embeddings=num_embeddings,
                                   use_gru=False)

        # sd = np.load('lstm.npy', allow_pickle=True).tolist()
        # lstm.set_state_dict(sd)

        state_dict = layer.state_dict()
        sd = OrderedDict()
        for key, value in state_dict.items():
            v = value.numpy()
            redirect_to_logger(key, value.shape, np.sum(v), np.mean(v), np.max(v), np.min(v))
            sd[key] = v

        np.save('att_gru_cell.npy', sd)

        hidden = fluid.dygraph.to_variable(x)
        inputs = fluid.dygraph.to_variable(y)
        char_onehots = fluid.dygraph.to_variable(z)
        (outputs, hidden), alpha = layer(hidden, inputs, char_onehots)
        # redirect_to_logger(len(outputs))
    return outputs.numpy()
    # return alpha.numpy()


def torch_grucell():
    np.random.seed(SEED)
    x = np.random.rand(1, 96).astype(np.float32)
    y = np.random.rand(1, 25, 192).astype(np.float32)
    z = np.zeros((1, 38)).astype(np.float32)
    z[0, 0] = 1
    hidden = torch.Tensor(x)
    inputs = torch.Tensor(y)
    char_onehots = torch.Tensor(z)

    layer = PTAttentionGRUCell(input_size=input_size,
                               hidden_size=hidden_size,
                               num_embeddings=num_embeddings,
                               use_gru=False)

    sd = np.load('att_gru_cell.npy', allow_pickle=True)
    sd = sd.tolist()

    for key, value in layer.state_dict().items():
        redirect_to_logger(key, value.shape)
        try:
            if key.endswith('.weight'):
                layer.state_dict()[key].copy_(torch.Tensor(sd[key].T))
            else:
                layer.state_dict()[key].copy_(torch.Tensor(sd[key]))
        except Exception as e:
            redirect_to_logger('pp: ', key, sd[key].shape)
            redirect_to_logger('pt: ', key, layer.state_dict()[key].shape)
            raise e


    (outputs, hidden), alpha = layer(hidden, inputs, char_onehots)
    return outputs.data.numpy()
    # return alpha.data.numpy()



if __name__ == '__main__':
    redirect_to_logger('==========paddle=================')
    a = paddle_grucell()
    redirect_to_logger(a.shape)
    redirect_to_logger('a: ', np.sum(a), np.mean(a), np.max(a), np.min(a))
    redirect_to_logger('===========pytorch================')
    b = torch_grucell()
    redirect_to_logger(b.shape)
    redirect_to_logger('b: ', np.sum(b), np.mean(b), np.max(b), np.min(b))