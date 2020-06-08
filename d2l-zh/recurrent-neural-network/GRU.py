"""
@Author: 	yshhuang@foxmail.com
@Date: 2020-06-08 15:33:56
@LastEditors: 	yshhuang@foxmail.com
@LastEditTime: 2020-06-08 16:01:05
@FilePath: /d2l-zh/recurrent-neural-network/GRU.py
"""
import d2lzh as d2l
from mxnet import nd
from mxnet.gluon import rnn


def get_params():
    '''
    @description: 
    @param {type} 
    @return: 
    '''
    def _one(shape):
        return nd.random.normal(scale=0.01, shape=shape, ctx=ctx)

    def _three():
        return (_one((num_inputs, num_hiddens)),
                _one((num_hiddens, num_hiddens)),
                nd.zeros(num_hiddens, ctx=ctx))

    W_xz, W_hz, b_z = _three()
    W_xr, W_hr, b_r = _three()
    W_xh, W_hh, b_h = _three()
    W_hq = _one((num_hiddens, num_outputs))
    b_q = nd.zeros(num_outputs, ctx=ctx)
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.attach_grad()
    return params


def init_gru_state(batch_size, num_hiddens, ctx):
    '''
    @description: 隐藏状态初始化函数
    @param {type} 
    @return: 
    '''
    return (nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx),)


def gru(inputs, state, params):
    """
    @description:门控循环单元 
    @param {type} 
    @return: 
    """
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = nd.sigmoid(nd.dot(X, W_xz)+nd.dot(H, W_hz)+b_z)
        R = nd.sigmoid(nd.dot(X, W_xr)+nd.dot(H, W_hr)+b_r)
        H_tilda = nd.tanh(nd.dot(X, W_xh)+nd.dot(R*H, W_hh)+b_h)
        H = Z*H+(1-Z)*H_tilda
        Y = nd.dot(H, W_hq)+b_q
        outputs.append(Y)
    return outputs, (H,)


if __name__ == "__main__":
    (corpus_indices, char_to_idx, idx_to_char,
     vocab_size) = d2l.load_data_jay_lyrics()
    num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
    ctx = d2l.try_gpu()

    num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
    pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']
    # d2l.train_and_predict_rnn(gru, get_params, init_gru_state, num_hiddens,
    #                                 vocab_size, ctx, corpus_indices, idx_to_char,
    #                                 char_to_idx, False, num_epochs, num_steps, lr,
    #                                 clipping_theta, batch_size, pred_period, pred_len,
    #                                 prefixes)
    gru_layer = rnn.GRU(num_hiddens)
    model = d2l.RNNModel(gru_layer, vocab_size)
    d2l.train_and_predict_rnn_gluon(model, num_hiddens, vocab_size, ctx,
                                    corpus_indices, idx_to_char, char_to_idx,
                                    num_epochs, num_steps, lr, clipping_theta,
                                    batch_size, pred_period, pred_len, prefixes)
