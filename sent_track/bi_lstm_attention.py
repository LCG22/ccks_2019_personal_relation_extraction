#! /usr/bin/env python
# coding: utf-8


"""

author: LCG22
date: 2019-09-20

content: 练习自己处理数据

update:
author: LCG22
date: 2019-10-30

content:
1、如果子类在继承后一定要实现的方法，可以在父类中指定metaclass为abc模块的ABCMeta
类，并在指定的方法上标准abc模块的@abcstractmethod来达到目的。
2、一旦定义了这样的父类，父类就不能实例化了，否则会抛出TypeError异常。
3、继承的子类如果没有实现@abcstractmethod标注的方法，在实例化使也会抛出TypeError异常。

原文链接：https://blog.csdn.net/caoxinjian423/article/details/83268457

"""

import pandas as pd
import numpy as np
import time
import jieba
import collections
import gensim
import json
import tensorflow as tf
import os
import datetime

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from configs import Config
from abc import ABCMeta, abstractmethod


class Vocab(object):

    def __init__(self):

        # EOS 是段结束
        # UNK 是未知，用于表示不在词汇表中的词
        # GO 是段开始，标记文本开始
        # PAD 是用于填充的，以使不同长度的文本的长度达到一致

        self.UNK = "<UNK>"
        self.EOS = "<EOS>"
        self.GO = "<GO>"
        self.PAD = "<PAD>"

        self.words_list = list()
        self.size = len([self.UNK, self.EOS, self.GO, self.PAD])
        self.length_of_words = len(self.words_list) + self.size
        self.length_of_count = self.size

        self.count = [tuple()]
        self.word2idx = {self.PAD: 0, self.UNK: 1, self.EOS: 2, self.GO: 3}
        self.idx2word = dict()

    def get_unk(self):

        return self.UNK

    def set_unk(self, unk):

        val = self.word2idx.get(self.get_unk(), 1)
        self.UNK = unk
        self.word2idx[self.get_unk()] = val

    def get_eos(self):

        return self.EOS

    def set_eos(self, eos):

        val = self.word2idx.get(self.get_eos(), 2)
        self.EOS = eos
        self.word2idx[self.get_eos()] = val

    def get_go(self):

        return self.GO

    def set_go(self, go):

        val = self.word2idx.get(self.get_go(), 2)
        self.GO = go
        self.word2idx[self.get_go()] = val

    def get_pad(self):

        return self.PAD

    def set_pad(self, pad):

        val = self.word2idx.get(self.get_pad(), 2)
        self.PAD = pad
        self.word2idx[self.get_pad()] = val

    def count_of_words(self):
        """
        统计所有单词的频数，并按频数从大到小进行排序，最后返回指定的个数的单词
        :return:
        """

        finally_count = [(self.PAD, 0), (self.UNK, 1), (self.GO, 2), (self.EOS, 3)]
        count = collections.Counter(self.words_list)
        self.length_of_count += len(count)
        self.size += self.size if self.size < self.length_of_count else self.length_of_count
        finally_count.extend(count.most_common(self.size - 2))

        self.count = finally_count

        return finally_count

    def build_vocab(self, words, size):
        """
        构建词典，以单词为键，映射的数字为值
        :param words:
        :param size:
        :return:
        """

        self.words_list = words
        self.length_of_words = len(words)
        self.size = size if size < self.length_of_words else self.length_of_words

        self.count_of_words()

        word2idx = {self.count[i_int][0]: i_int for i_int in range(len(self.count))}

        self.word2idx.update(word2idx)

        self.idx2word = self.reverse_vocab()

        return self.word2idx, self.idx2word

    def build_vocab_by_increment(self, words, size):
        """
        增量创建词典，主要是为了应对文本数据量大，电脑内存不足的场景
        :param words:
        :param size:
        :return:
        """

        if not self.word2idx:
            tmp_top_n = [(self.PAD, 0), (self.UNK, 1), (self.GO, 2), (self.EOS, 3)]
        else:
            tmp_top_n = []

        all_word_cnt = 0
        for word in words:

            all_word_cnt += 1

            self.word2idx[word] = 1 + self.word2idx.get(word, 0)

            if all_word_cnt % 10000 == 0:
                print("已处理了 {} 个词".format(all_word_cnt))

        self.length_of_count += all_word_cnt

        counter = collections.Counter(self.word2idx)

        self.count = counter.most_common(size - len(tmp_top_n))
        tmp_top_n.extend(self.count)

        self.count = tmp_top_n

        word2idx = {self.count[i_int][0]: i_int for i_int in range(len(self.count))}

        self.word2idx = word2idx

        self.idx2word = self.reverse_vocab()

        return self.word2idx, self.idx2word

    def reverse_vocab(self):
        """
        翻转词典，翻转为以映射的数字为键，单词为值
        :return:
        """

        reversed_vocab = {self.word2idx[key]: key for key in self.word2idx.keys()}

        self.idx2word = reversed_vocab

        return reversed_vocab

    def get_unk_index(self):

        index = self.word2idx.get(self.UNK)

        return index

    def get_word_to_index(self, word):

        return self.word2idx.get(word, self.get_unk_index())

    def get_index_to_word(self, index):

        return self.idx2word.get(index, self.UNK)

    def get_words_to_index(self, words):

        indexs = []
        for word in words:
            indexs.append(self.get_word_to_index(word))

        return indexs

    def get_indexs_to_word(self, indexs):

        words = []
        for index in indexs:
            words.append(self.get_index_to_word(index))

        return words

    def read_vocab(self, path):

        vocab = np.load(file=path).item()

        self.word2idx = vocab

        print("数据已经读取完毕！")

        return vocab

    def read_reversed_vocab(self, path):

        reversed_vocab = np.load(file=path).item()

        self.idx2word = reversed_vocab

        print("翻转字典数据已经读取完毕！")

        return reversed_vocab

    def save_vocab(self, path):

        np.save(path, self.word2idx)

        print("字典数据已经保存完毕！")

    def save_reversed_vocab(self, path):

        np.save(path, self.idx2word)

        print("数据已经保存完毕！")

    def get_top_rank_words(self, n, length=1):

        if length == 2:
            count = self.count
        else:
            count = list(filter(lambda x: True if len(x[0]) >= length else False, self.count))

        words = count[: n]

        return words

    def get_tail_rank_words(self, n, length=2):

        if length == 2:
            count = self.count
        else:
            count = list(filter(lambda x: True if len(x[0]) >= length else False, self.count))

        words = count[n:]

        return words


class BaseProcessingData(metaclass=ABCMeta):

    def __init__(self, config, train):

        self.config = config
        # 是否是训练集状态
        self.train = train
        self.label_name = config.label_name
        self.sentence_name = config.sentence_name
        self.stop_words_path = self.config.stop_words_path
        self.low_frequency_word_threshold = self.config.low_frequency_word_threshold
        self.size_of_word = self.config.size_of_word
        self.batch_size = config.batch_size
        self.word_embedding_path = self.config.word_embedding_path
        self.word_embedding_size = config.word_embedding_size

        self.max_sequence_length = config.max_sequence_length
        self.num_classes = config.num_classes

        self.unique_labels = list()
        self.word2idx = dict()
        self.idx2word = dict()
        self.label2idx = dict()
        self.idx2label = dict()
        self.word_embedding = dict()

        # 用户自定义的分词词典
        self.user_lexicon = config.user_lexicon
        # 分词器，必须要实现 cut 的生成器功能，以及支持用户自定义分词词典
        self.segmenter = self.gen_segmenter(user_lexicon=self.user_lexicon)

        # 词典类，使用词来构建词典
        self.vocab = Vocab()

        self.stop_words = self.gen_stop_words(path=self.stop_words_path)
        self.label2idx_path = self.config.label2idx_path
        self.word2idx_path = self.config.word2idx_path

    def gen_stop_words(self, path):

        if path:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                stop_words_list = content.splitlines()
                stop_words = dict(zip(stop_words_list, list(range(len(stop_words_list)))))
        else:
            stop_words = dict()

        return stop_words

    def gen_segmenter(self, user_lexicon):

        segmenter = jieba
        if user_lexicon:
            segmenter.load_userdict(user_lexicon)

        return segmenter

    def read_data(self, path, chunker=False, sep=",", error_bad_lines=True):

        print("开始读取数据。。。")
        if chunker:
            temp = pd.read_csv(path, engine="python", iterator=True, sep=sep, error_bad_lines=error_bad_lines)
            loop = True
            chunk_size = 10000
            chunks = []
            while loop:
                try:
                    chunk = temp.get_chunk(chunk_size)
                    chunks.append(chunk)
                except StopIteration as e:
                    loop = False
            df = pd.concat(chunks, ignore_index=True, axis=0)
        else:
            df = pd.read_csv(path, sep=sep, error_bad_lines=error_bad_lines)
        print("数据读取成功！")
        print("数据维度为：{}".format(df.shape))

        return df

    def reset_columns(self, df, **kwargs):

        df.columns = kwargs["columns"]

        return df

    def as_type(self, series, data_type):

        new_series = series.astype(data_type)

        return new_series

    def clean_str(self, series):

        new_series = series.apply(str.strip)

        return new_series

    def segment_word(self, sentence):

        new_sentence = list(self.segmenter.cut(sentence=sentence))

        return new_sentence

    def batch_segment_word(self, series):

        new_series = series.apply(self.segment_word)

        return new_series

    def gen_all_words(self, series):

        all_words = list()
        series.apply(all_words.extend)

        return all_words

    def build_vocab(self, words, size):

        word2idx, idx2word = self.vocab.build_vocab(words=words, size=size)
        self.word2idx = word2idx
        self.idx2word = idx2word

        return word2idx, idx2word

    def get_word2idx_dict(self):

        return self.word2idx

    def get_idx2word_dict(self):

        return self.idx2word

    def get_label2idx_dict(self):

        return self.label2idx

    def get_idx2label_dict(self):

        return self.idx2label

    def gen_label2idx_dict(self, labels):

        unique_labels = list(set(labels))

        label2idx = dict(zip(unique_labels, list(range(len(unique_labels)))))
        self.unique_labels = unique_labels
        self.label2idx = label2idx

        return label2idx

    def get_unique_labels(self):

        return self.unique_labels

    def gen_word2idx(self, words):

        ids = [self.word2idx.get(word, self.word2idx[self.vocab.get_unk()]) for word in words]

        return ids

    def batch_gen_word2idx(self, words):

        ids = list()
        for sub_words in words:
            sub_ids = self.gen_word2idx(words=sub_words)
            ids.append(sub_ids)

        return ids

    def gen_label2idx(self, labels):

        ids = [self.label2idx[label] for label in labels]

        return ids

    def gen_word_embedding(self, words, path=None, binary=True):

        if path:
            word_embedding = gensim.models.KeyedVectors.load_word2vec_format(path, binary=binary)
        else:
            word_embedding = dict()

        new_word_embedding = list()
        for word in words:
            try:
                word_vec = word_embedding.wv[word]
                new_word_embedding.append(word_vec)
            except Exception as e:
                print("{} 不在词向量中，故取随机值".format(word))
                new_word_embedding.append(np.random.randn(self.word_embedding_size))

        new_word_embedding = np.array(new_word_embedding)
        self.word_embedding = new_word_embedding

        return new_word_embedding

    def padding(self, sequence, pad):

        sequence_length = len(sequence)
        if sequence_length > self.max_sequence_length:
            new_sequence = sequence[: self.max_sequence_length]
        else:
            padding_words = [pad] * (self.max_sequence_length - sequence_length)
            new_sequence = sequence[:]
            new_sequence.extend(padding_words)

        return new_sequence

    def batch_padding(self, sequences, pad):

        new_sequences = list()
        for sequence in sequences:
            new_sequence = self.padding(sequence=sequence, pad=pad)
            new_sequences.append(new_sequence)

        return new_sequences

    def gen_train_eval_data(self, x, y, rate=None):

        index = int(len(x) * rate)

        x = np.array(x)
        y = np.array(y)

        train_x = x[: index]
        eval_x = x[index:]

        train_y = y[: index]
        eval_y = y[index:]

        return train_x, train_y, eval_x, eval_y

    def save_data_with_json(self, data, path):
        """
        保存数据
        :param data:
        :param path:
        :return:
        """

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
            print("数据保存结束！")

    @abstractmethod
    def get_output(self):

        raise()

        # if self.train:
        #     df = self.read_data(path=self.config.train_data_path, chunker=self.config.read_chunk)
        # else:
        #     df = self.read_data(path=self.config.test_data_path, chunker=self.config.read_chunk)
        #
        # # tmp
        # # df = df[: 10]
        #
        # if self.num_classes == 2:
        #     df = self.reset_columns(df, columns=["id", "x", "sentiment"])
        #     labels = df["sentiment"]
        # elif self.num_classes > 2:
        #     df = self.reset_columns(df, columns=["id", "x", "rate"])
        #     labels = df["rate"]
        #
        # labels = self.as_type(labels, int)
        # labels = labels.tolist()
        #
        # x = df["x"]
        #
        # cleaned_x = self.clean_str(series=x)
        # segment_x = self.batch_segment_word(series=cleaned_x)
        # all_words = self.gen_all_words(series=segment_x)
        # word2idx, idx2word = self.build_vocab(words=all_words, size=self.size_of_word)
        # label2idx = self.gen_label2idx_dict(labels=labels)
        # x_ids = self.batch_gen_word2idx(words=segment_x)
        # label_ids = self.gen_label2idx(labels=labels)
        # paded_x = self.batch_padding(sequences=x_ids, pad=word2idx[self.vocab.get_pad()])
        # word_embedding = self.gen_word_embedding(words=list(word2idx.keys()), path=self.word_embedding_path)
        #
        # train_x, train_y, eval_x, eval_y = self.gen_train_eval_data(x=paded_x, y=label_ids,
        #                                                             rate=self.config.train_data_rate)
        #
        # self.save_data_with_json(data=word2idx, path=self.word2idx_path)
        # self.save_data_with_json(data=label2idx, path=self.label2idx_path)
        #
        # return train_x, train_y, eval_x, eval_y


class ProcessingData(BaseProcessingData):

    def __init__(self, config, train):
        super(ProcessingData, self).__init__(config=config, train=train)

    def get_output(self):

        if self.train:
            df = self.read_data(path=self.config.train_data_path, chunker=self.config.read_chunk,
                                sep=self.config.read_csv_sep)
        else:
            df = self.read_data(path=self.config.test_data_path, chunker=self.config.read_chunk,
                                sep=self.config.read_csv_sep)

        df = df[df[self.label_name].notnull()]

        labels = df[self.label_name]
        labels = self.as_type(labels, int)
        labels = labels.tolist()

        all_words = []
        all_segment_x = []
        if len(self.sentence_name) == 1:
            x = df[self.sentence_name[0]]

            cleaned_x = self.clean_str(series=x)
            segment_x = self.batch_segment_word(series=cleaned_x)
            tmp_all_words = self.gen_all_words(series=segment_x)
            all_words.extend(tmp_all_words)
            all_segment_x.extend(segment_x)
        else:
            for name in self.sentence_name:
                x = df[name]

                cleaned_x = self.clean_str(series=x)
                segment_x = self.batch_segment_word(series=cleaned_x)
                tmp_all_words = self.gen_all_words(series=segment_x)
                all_words.extend(tmp_all_words)
                all_segment_x.append(segment_x)

        word2idx, idx2word = self.build_vocab(words=all_words, size=self.size_of_word)
        label2idx = self.gen_label2idx_dict(labels=labels)
        label_ids = self.gen_label2idx(labels=labels)

        word_embedding = self.gen_word_embedding(words=list(word2idx.keys()), path=self.word_embedding_path)
        self.word_embedding = word_embedding
        all_paded_x = []
        for segment_x in all_segment_x:
            x_ids = self.batch_gen_word2idx(words=segment_x)
            paded_x = self.batch_padding(sequences=x_ids, pad=word2idx[self.vocab.get_pad()])
            all_paded_x.append(paded_x)

        # 将元素按行合并
        paded_x = np.concatenate(all_paded_x, axis=1)

        train_x, train_y, eval_x, eval_y = self.gen_train_eval_data(x=paded_x, y=label_ids,
                                                                    rate=self.config.train_data_rate)

        self.save_data_with_json(data=word2idx, path=self.word2idx_path)
        self.save_data_with_json(data=label2idx, path=self.label2idx_path)

        return train_x, train_y, eval_x, eval_y


def next_batch(x, y, batch_size):
    """
    生成batch数据集，用生成器的方式输出
    """

    perm = np.arange(len(x))
    np.random.shuffle(perm)
    x = x[perm]
    y = y[perm]

    batches_num = len(x) // batch_size

    for i in range(batches_num):
        start = i * batch_size
        end = start + batch_size
        batch_x = np.array(x[start: end], dtype="int64")
        batch_y = np.array(y[start: end], dtype="float32")

        yield batch_x, batch_y


# In[6]:


class BiLSTM(object):

    def __init__(self, config, word_embedding=None):

        self.config = config

        self.x = None
        self.y = None
        self.dropout_keep_prob = None

        self.word_embedding = word_embedding
        self.vocab_size = config.size_of_word
        self.embedding_size = config.word_embedding_size
        self.embedding = None
        self.hidden_sizes = config.model.hidden_sizes

        self.num_classes = config.num_classes
        self.l2_reg_lambda = config.model.l2_reg_lambda
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
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)

            if l2_reg_lambda and l2_loss:
                loss = tf.reduce_mean(loss) + l2_reg_lambda * l2_loss
            else:
                loss = tf.reduce_mean(loss)

        self.loss = loss

        return loss

    def attention(self, H):

        # 获得最后一层 LSTM 神经元的个数
        hidden_size = config.model.hidden_sizes[-1]

        # 初始化一个权重向量，是可训练的参数
        W = tf.Variable(tf.random_normal([hidden_size], stddev=0.1))

        # 对 LSTM 的输出使用激活函数，进行非线性转换
        M = tf.tanh(H)

        # 对 W 和 M 做矩阵运算
        # M = [batch_size, timeStep, hidden_size]，将 M 转换为 [batch_size * timeStep, hidden_size]
        # newM = [batch_size, timeStep, 1]，每一个时间步的输出由向量转换成数字
        newM = tf.matmul(tf.reshape(M, [-1, hidden_size]), tf.reshape(W, [-1, 1]))

        # 对 newM 做维度转换成 [batch_size, timeStep]
        restoreM = tf.reshape(newM, [-1, config.max_sequence_length])

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


# In[7]:


"""
定义各类性能指标
"""


def mean(item: list) -> float:
    """
    计算列表中元素的平均值
    :param item: 列表对象
    :return:
    """
    res = sum(item) / len(item) if len(item) > 0 else 0
    return res


def accuracy(pred_y, true_y):
    """
    计算二类和多类的准确率
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :return:
    """
    if isinstance(pred_y[0], list):
        pred_y = [item[0] for item in pred_y]
    corr = 0
    for i in range(len(pred_y)):
        if pred_y[i] == true_y[i]:
            corr += 1
    acc = corr / len(pred_y) if len(pred_y) > 0 else 0
    return acc


def binary_precision(pred_y, true_y, positive=1):
    """
    二类的精确率计算
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :param positive: 正例的索引表示
    :return:
    """
    corr = 0
    pred_corr = 0
    for i in range(len(pred_y)):
        if pred_y[i] == positive:
            pred_corr += 1
            if pred_y[i] == true_y[i]:
                corr += 1

    prec = corr / pred_corr if pred_corr > 0 else 0
    return prec


def binary_recall(pred_y, true_y, positive=1):
    """
    二类的召回率
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :param positive: 正例的索引表示
    :return:
    """
    corr = 0
    true_corr = 0
    for i in range(len(pred_y)):
        if true_y[i] == positive:
            true_corr += 1
            if pred_y[i] == true_y[i]:
                corr += 1

    rec = corr / true_corr if true_corr > 0 else 0
    return rec


def binary_f_beta(pred_y, true_y, beta=1.0, positive=1):
    """
    二类的f beta值
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :param beta: beta值
    :param positive: 正例的索引表示
    :return:
    """
    precision = binary_precision(pred_y, true_y, positive)
    recall = binary_recall(pred_y, true_y, positive)
    try:
        f_b = (1 + beta * beta) * precision * recall / (beta * beta * precision + recall)
    except:
        f_b = 0
    return f_b


def multi_precision(pred_y, true_y, labels):
    """
    多类的精确率
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :param labels: 标签列表
    :return:
    """
    if isinstance(pred_y[0], list):
        pred_y = [item[0] for item in pred_y]

    precisions = [binary_precision(pred_y, true_y, label) for label in labels]
    prec = mean(precisions)
    return prec


def multi_recall(pred_y, true_y, labels):
    """
    多类的召回率
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :param labels: 标签列表
    :return:
    """
    if isinstance(pred_y[0], list):
        pred_y = [item[0] for item in pred_y]

    recalls = [binary_recall(pred_y, true_y, label) for label in labels]
    rec = mean(recalls)
    return rec


def multi_f_beta(pred_y, true_y, labels, beta=1.0):
    """
    多类的f beta值
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :param labels: 标签列表
    :param beta: beta值
    :return:
    """
    if isinstance(pred_y[0], list):
        pred_y = [item[0] for item in pred_y]

    f_betas = [binary_f_beta(pred_y, true_y, beta, label) for label in labels]
    f_beta = mean(f_betas)
    return f_beta


def get_binary_metrics(pred_y, true_y, f_beta=1.0):
    """
    得到二分类的性能指标
    :param pred_y:
    :param true_y:
    :param f_beta:
    :return:
    """
    acc = accuracy(pred_y, true_y)
    recall = binary_recall(pred_y, true_y)
    precision = binary_precision(pred_y, true_y)
    f_beta = binary_f_beta(pred_y, true_y, f_beta)
    return acc, recall, precision, f_beta


def get_multi_metrics(pred_y, true_y, labels, f_beta=1.0):
    """
    得到多分类的性能指标
    :param pred_y:
    :param true_y:
    :param labels:
    :param f_beta:
    :return:
    """
    acc = accuracy(pred_y, true_y)
    recall = multi_recall(pred_y, true_y, labels)
    precision = multi_precision(pred_y, true_y, labels)
    f_beta = multi_f_beta(pred_y, true_y, labels, f_beta)
    return acc, recall, precision, f_beta


if __name__ == '__main__':

    config = Config()
    processing = ProcessingData(config=config, train=True)
    train_x, train_y, eval_x, eval_y = processing.get_output()
    # 生成训练集和验证集
    train_x = train_x
    train_y = train_y
    eval_x = eval_x
    eval_y = eval_y

    word_embedding = processing.word_embedding
    unique_labels = processing.unique_labels

    # 定义计算图
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_conf.gpu_options.allow_growth = True
        session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9  # 配置gpu占用率

        sess = tf.Session(config=session_conf)

        # 定义会话
        with sess.as_default():
            lstm = BiLSTM(config, word_embedding)
            lstm.gen_model()

            global_step = tf.Variable(0, name="global_step", trainable=False)
            # 定义优化函数，传入学习速率参数
            optimizer = tf.train.AdamOptimizer(config.training.learning_rate)
            # 计算梯度,得到梯度和变量
            grads_and_vars = optimizer.compute_gradients(lstm.loss)
            # 将梯度应用到变量下，生成训练器
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # 用summary绘制tensorBoard
            # gradSummaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))

            out_dir = os.path.abspath(os.path.join(os.path.curdir, "summarys"))
            print("Writing to {}\n".format(out_dir))

            loss_summary = tf.summary.scalar("loss", lstm.loss)
            summary_op = tf.summary.merge_all()

            train_summary_dir = os.path.join(out_dir, "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            eval_summary_dir = os.path.join(out_dir, "eval")
            eval_summary_writer = tf.summary.FileWriter(eval_summary_dir, sess.graph)

            # 初始化所有变量
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

            # 保存模型的一种方式，保存为pb文件
            save_model_path = config.save_model_path
            if os.path.exists(save_model_path):
                try:
                    os.rmdir(save_model_path)
                except OSError as e:
                    # 如果目录非空，则使用如下方法删除
                    import shutil

                    shutil.rmtree(save_model_path)
            builder = tf.saved_model.builder.SavedModelBuilder(save_model_path)

            sess.run(tf.global_variables_initializer())


            def train_step(batch_x, batch_y):
                """
                训练函数
                """
                feed_dict = {
                    lstm.x: batch_x,
                    lstm.y: batch_y,
                    lstm.dropout_keep_prob: config.model.dropout_keep_prob
                }
                _, summary, step, loss, predictions = sess.run(
                    [train_op, summary_op, global_step, lstm.loss, lstm.predictions],
                    feed_dict)

                if config.num_classes == 2:
                    acc, recall, prec, f_beta = get_binary_metrics(pred_y=predictions, true_y=batch_y)

                elif config.num_classes > 2:
                    acc, recall, prec, f_beta = get_multi_metrics(pred_y=predictions, true_y=batch_y,
                                                                  labels=unique_labels)

                train_summary_writer.add_summary(summary, step)

                return loss, acc, prec, recall, f_beta


            def dev_step(batch_x, batch_y):
                """
                验证函数
                """
                feed_dict = {
                    lstm.x: batch_x,
                    lstm.y: batch_y,
                    lstm.dropout_keep_prob: 1.0
                }
                summary, step, loss, predictions = sess.run(
                    [summary_op, global_step, lstm.loss, lstm.predictions],
                    feed_dict)

                if config.num_classes == 2:

                    acc, precision, recall, f_beta = get_binary_metrics(pred_y=predictions, true_y=batch_y)
                elif config.num_classes > 2:
                    acc, precision, recall, f_beta = get_multi_metrics(pred_y=predictions, true_y=batch_y,
                                                                       labels=unique_labels)

                eval_summary_writer.add_summary(summary, step)

                return loss, acc, precision, recall, f_beta


            for i in range(config.training.epoches):
                # 训练模型
                print("start training model")
                for batch_x, batch_y in next_batch(train_x, train_y, config.batch_size):
                    loss, acc, prec, recall, f_beta = train_step(batch_x, batch_y)

                    current_step = tf.train.global_step(sess, global_step)
                    time_str = datetime.datetime.now().isoformat()
                    print(
                        "current time: {}, train: step: {}, loss: {}, acc: {}, recall: {}, precision: {}, f_beta: {}".format(
                            time_str, current_step, loss, acc, recall, prec, f_beta))
                    if current_step % config.training.evaluate_every == 0:
                        print("\nEvaluation:")

                        losses = []
                        accs = []
                        f_betas = []
                        precisions = []
                        recalls = []

                        for eval_batch_x, eval_batch_y in next_batch(eval_x, eval_y, config.batch_size):
                            loss, acc, precision, recall, f_beta = dev_step(eval_batch_x, eval_batch_y)
                            losses.append(loss)
                            accs.append(acc)
                            f_betas.append(f_beta)
                            precisions.append(precision)
                            recalls.append(recall)

                        time_str = datetime.datetime.now().isoformat()
                        print("current time: {}, step: {}, loss: {}, acc: {},precision: {}, recall: {}, f_beta: {}".format(
                            time_str,
                            current_step,
                            mean(losses),
                            mean(accs),
                            mean(precisions),
                            mean(recalls),
                            mean(f_betas)))

                    if current_step % config.training.checkpoint_every == 0:
                        # 保存模型的另一种方法，保存checkpoint文件
                        path = saver.save(sess, "model/Bi-LSTM/model/my-model", global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))

            inputs = {"input_x": tf.saved_model.utils.build_tensor_info(lstm.x),
                      "keep_prob": tf.saved_model.utils.build_tensor_info(lstm.dropout_keep_prob)}

            outputs = {"predictions": tf.saved_model.utils.build_tensor_info(lstm.predictions)}

            prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
                inputs=inputs, outputs=outputs, method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
            legacy_init_op = tf.group(tf.tables_initializer(), name="legacy_init_op")
            builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
                                                 signature_def_map={"predict": prediction_signature},
                                                 legacy_init_op=legacy_init_op)

            builder.save()

