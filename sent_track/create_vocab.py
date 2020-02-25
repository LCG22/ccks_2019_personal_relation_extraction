#! /usr/bin/env python
# coding: utf-8


import collections
import numpy as np


class Vocab(object):

    def __init__(self):

        # EOS 是段结束
        # UNK 是未知，用于表示不在词汇表中的词
        # GO 是段开始，标记文本开始
        # PAD 是用于填充的，以使不同长度的文本的长度达到一致
        # SPACING 是用于表明 SPACING 符号左右两边的元素原本是隔开的，但是因为某些原因而不得不合并在一起

        self.UNK = "<UNK>"
        self.EOS = "<EOS>"
        self.GO = "<GO>"
        self.PAD = "<PAD>"
        self.SPACING = "<SPACING>"

        self.words_list = list()
        self.size = len([self.UNK, self.EOS, self.GO, self.PAD, self.SPACING])
        self.length_of_words = len(self.words_list) + self.size
        self.length_of_count = self.size

        self.count = [tuple()]
        self.word2idx = {self.PAD: 0, self.UNK: 1, self.EOS: 2, self.GO: 3, self.SPACING: 4}
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

        val = self.word2idx.get(self.get_go(), 3)
        self.GO = go
        self.word2idx[self.get_go()] = val

    def get_pad(self):

        return self.PAD

    def set_pad(self, pad):

        val = self.word2idx.get(self.get_pad(), 0)
        self.PAD = pad
        self.word2idx[self.get_pad()] = val

    def get_spacing(self):

        return self.SPACING

    def set_spacing(self, spacing):

        val = self.word2idx.get(self.get_spacing(), 4)
        self.SPACING = spacing
        self.word2idx[self.get_spacing()] = val

    def count_of_words(self):
        """
        统计所有单词的频数，并按频数从大到小进行排序，最后返回指定的个数的单词
        :return:
        """

        finally_count = [(self.PAD, 0), (self.UNK, 1), (self.GO, 2), (self.EOS, 3), (self.SPACING, 4)]
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
            tmp_top_n = [(self.PAD, 0), (self.UNK, 1), (self.GO, 2), (self.EOS, 3), (self.SPACING, 4)]
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