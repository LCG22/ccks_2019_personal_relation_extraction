#! /usr/bin/env python
# coding: utf-8


"""

author: LCG22
date: 2020-02-22

content: Bi-LSTM-Attention 模型

"""

import tensorflow as tf


class BiLSTM(object):

    def __init__(self, config, word_embedding=None):

        self.config = config

        self.x = None
        self.y = None
        self.dropout_keep_prob = None

        self.word_embedding = word_embedding
        self.vocab_size = config.size_of_word
        self.embedding_size = config.embedding_size
        self.embedding = None
        self.hidden_sizes = config.hidden_sizes

        self.num_classes = config.num_classes
        self.l2_reg_lambda = config.l2_reg_lambda
        self.l2_loss = 0.0

        self.predictions = None
        self.loss = None

    def gen_inputs(self):

        with tf.name_scope("Inputs"):
            x = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_x")
            y = tf.placeholder(dtype=tf.int32, shape=[None], name="y")
            dropout_keep_prob = tf.placeholder(dtype=tf.float32, shape=None, name="dropout_keep_prob")

        self.x = x
        self.y = y
        self.dropout_keep_prob = dropout_keep_prob

        return x, y, dropout_keep_prob

    def gen_word_embedding(self, ids, vocab_size=None, embedding_size=None, word_embedding=None):

        with tf.variable_scope("Embedding", reuse=tf.AUTO_REUSE):
            if word_embedding is not None:
                word_embedding = tf.Variable(tf.cast(word_embedding, dtype=tf.float32), trainable=True,
                                             name="word_embedding")
                self.word_embedding = word_embedding
            else:
                word_embedding = tf.Variable(tf.truncated_normal(shape=[vocab_size, embedding_size],
                                                                 dtype=tf.float32))
                self.word_embedding = word_embedding

            embedding = tf.nn.embedding_lookup(word_embedding, ids, name="embedding")
        self.embedding = embedding

        return embedding

    def gen_bilstm(self, hidden_sizes, dropout_keep_prob, inputs):

        with tf.name_scope("Bi-LSTM"):
            for idx, hidden_size in enumerate(hidden_sizes):
                with tf.name_scope("Bi-LSTM_{}".format(idx)):
                    fw_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                        cell=tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True),
                        output_keep_prob=dropout_keep_prob)
                    bw_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                        cell=tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True),
                        output_keep_prob=dropout_keep_prob)

                    outputs, states = tf.nn.bidirectional_dynamic_rnn(fw_lstm_cell, bw_lstm_cell, inputs,
                                                                      dtype=tf.float32,
                                                                      scope="bi-lstm_{}".format(idx))

                    inputs = tf.concat(outputs, -1)

        outputs = tf.split(inputs, num_or_size_splits=2, axis=-1)

        with tf.name_scope("Attention"):
            output = outputs[0] + outputs[1]

            att_output = self.attention(output)

        return att_output

    def gen_model_output(self, inputs, output_size, num_classes):

        if num_classes == 2:
            num_classes -= 1

        with tf.name_scope("Output"):
            weights = tf.get_variable(name="weights", shape=[output_size, num_classes], dtype=tf.float32,
                                      initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable(name="biases", shape=[num_classes], dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer())

            self.l2_loss += tf.nn.l2_loss(weights)
            self.l2_loss += tf.nn.l2_loss(biases)

            logits = tf.nn.xw_plus_b(x=inputs, weights=weights, biases=biases, name="logits")

            if self.num_classes == 2:
                predictions = tf.cast(tf.greater_equal(logits, 0.0), dtype=tf.float32, name="predictions")
            elif self.num_classes > 2:
                predictions = tf.argmax(logits, axis=-1, name="predictions")

        self.predictions = predictions

        return logits, predictions

    def gen_loss(self, num_classes, logits, labels, l2_reg_lambda=None, l2_loss=None):

        with tf.name_scope("Loss"):
            if num_classes == 2:
                loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=logits, labels=tf.cast(tf.reshape(labels, shape=[-1, 1]), dtype=tf.float32))
            elif num_classes > 2:
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.clip_by_value(logits, 1e-10, 100.0),
                                                                      labels=labels)

            if l2_reg_lambda and (l2_loss is not None):
                loss = tf.reduce_mean(loss) + l2_reg_lambda * l2_loss
            else:
                loss = tf.reduce_mean(loss)

        self.loss = loss

        return loss

    def attention(self, H):

        # 获得最后一层 LSTM 神经元的个数
        hidden_size = self.config.hidden_sizes[-1]

        # 初始化一个权重向量，是可训练的参数
        W = tf.Variable(tf.random_normal([hidden_size], stddev=0.1))

        # 对 LSTM 的输出使用激活函数，进行非线性转换
        M = tf.tanh(H)

        # 对 W 和 M 做矩阵运算
        # M = [batch_size, timeStep, hidden_size]，将 M 转换为 [batch_size * timeStep, hidden_size]
        # newM = [batch_size, timeStep, 1]，每一个时间步的输出由向量转换成数字
        newM = tf.matmul(tf.reshape(M, [-1, hidden_size]), tf.reshape(W, [-1, 1]))

        # 对 newM 做维度转换成 [batch_size, timeStep]
        restoreM = tf.reshape(newM, [-1, self.config.max_sequence_length])

        # 用 softmax 做归一化处理 [batch_size, timeStep]
        self.alpha = tf.nn.softmax(restoreM)

        # 利用求得的 alpha 的值进行加权求和，用矩阵运算直接操作
        r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(self.alpha, [-1, self.config.max_sequence_length, 1]))

        # 将三维压缩成二维 [batch_size, hidden_size]
        suqueezeR = tf.reshape(r, [-1, hidden_size])

        sentenceRepren = tf.tanh(suqueezeR)

        # 对 Attention 的输出可以做 dropout 处理
        output = tf.nn.dropout(sentenceRepren, keep_prob=self.dropout_keep_prob)

        return output

    def gen_model(self):

        x, y, dropout_keep_prob = self.gen_inputs()
        embedding = self.gen_word_embedding(ids=x, vocab_size=self.vocab_size, embedding_size=self.embedding_size,
                                            word_embedding=self.word_embedding)
        bi_lstm_model_output = self.gen_bilstm(hidden_sizes=self.hidden_sizes, dropout_keep_prob=self.dropout_keep_prob,
                                               inputs=embedding)

        logits, predictions = self.gen_model_output(inputs=bi_lstm_model_output, output_size=self.hidden_sizes[-1],
                                                    num_classes=self.num_classes)

        loss = self.gen_loss(num_classes=self.num_classes, logits=logits, labels=y, l2_reg_lambda=self.l2_reg_lambda,
                             l2_loss=self.l2_loss)

        return loss
