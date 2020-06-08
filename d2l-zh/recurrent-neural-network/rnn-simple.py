# encoding:utf-8
"""
@Time    : 2020-05-25 18:51
@Author  : yshhuang@foxmail.com
@File    : rnn-simple.py
@Software: PyCharm
"""
import d2lzh as d2l
import math
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn, rnn
import time


class RNNModel(nn.Block):
    """
    自定义循环神经网络
    """

    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.dense = nn.Dense(vocab_size)

    def forward(self, inputs, state):
        X = nd.one_hot(inputs.T, self.vocab_size)
        Y, state = self.rnn(X, state)
        output = self.dense(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)


def predict_rnn_gluon(prefix, num_chars, model, vocab_size, ctx, idx_to_char, char_to_idx):
    """
    预测函数
    """
    state = model.begin_state(batch_size=1, ctx=ctx)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars+len(prefix)-1):
        X = nd.array([output[-1]], ctx=ctx).reshape((1, 1))
        (Y, state) = model(X, state)
        if t < len(prefix)-1:
            output.append(char_to_idx[prefix[t+1]])
        else:
            output.append(int(Y.argmax(axis=1).asscalar()))
    return ''.join([idx_to_char[i] for i in output])


def train_and_predict_rnn_gluon(model, num_hiddens, vocab_size, ctx, corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta, batch_size, pred_period, pred_len, prefixes):
    """
    训练函数
    """
    loss = gloss.SoftmaxCrossEntropyLoss()
    model.initialize(ctx=ctx, force_reinit=True, init=init.Normal(0.01))
    trainer = gluon.Trainer(model.collect_params(), 'sgd', {
                            'learning_rate': lr, 'momentum': 0, 'wd': 0})
    for epoch in range(num_epochs):
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = d2l.data_iter_consecutive(
            corpus_indices, batch_size, num_steps, ctx)
        state = model.begin_state(batch_size=batch_size, ctx=ctx)
        for X, Y in data_iter:
            for s in state:
                s.detach()
            with autograd.record():
                (output, state) = model(X, state)
                y = Y.T.reshape((-1,))
                l = loss(output, y).mean()
            l.backward()
            # 梯度裁剪
            params = [p.data() for p in model.collect_params().values()]
            d2l.grad_clipping(params, clipping_theta, ctx)
            trainer.step(1)
            l_sum += l.asscalar()*y.size
            n += y.size
        if (epoch+1) % pred_period == 0:
            print('epoch %d, perplexity %f,time %.2f sec' %
                  (epoch+1, math.exp(l_sum/n), time.time()-start))
            for prefix in prefixes:
                print(' -', predict_rnn_gluon(prefix,
                                              pred_len, model, vocab_size, ctx, idx_to_char, char_to_idx))


if __name__ == "__main__":
    (corpus_indices, char_to_idx, idx_to_char, vocab_size) \
        = d2l.load_data_jay_lyrics()
    num_hiddens = 256
    rnn_layer = rnn.RNN(num_hiddens)
    rnn_layer.initialize()
    batch_size = 2
    state = rnn_layer.begin_state(batch_size=batch_size)
    # print(state[0].shape)
    num_steps = 35
    X = nd.random.uniform(shape=(num_steps, batch_size, vocab_size))
    T, state_new = rnn_layer(X, state)
    # print(T.shape, len(state_new), state_new[0].shape)
    ctx = d2l.try_gpu()
    model = RNNModel(rnn_layer, vocab_size)
    model.initialize(force_reinit=True, ctx=ctx)
    print(predict_rnn_gluon('分开', 10, model, vocab_size,
                            ctx, idx_to_char, char_to_idx))

    num_epochs, batch_size, lr, clipping_theta = 250, 32, 1e2, 1e-2
    pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
    train_and_predict_rnn_gluon(model, num_hiddens, vocab_size, ctx,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes)
