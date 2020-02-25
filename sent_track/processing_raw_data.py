#! /usr/bin/env python
# coding: utf-8


"""

author: LCG22
date: 2020-02-22

content: 处理原始数据，将多个数据集合并成为一个数据集
"""


import pandas as pd
import numpy as np


class ProcessingRawData(object):

    def __init__(self, sent_path, sent_relation_path, output_file_path, max_sequence_length, is_test=False):

        self.sent_path = sent_path
        self.sent_relation_path = sent_relation_path
        self.max_sequence_length = max_sequence_length
        self.sent_col_names = ["sent_id", "per_1", "per_2", "sent"]
        self.sent_relation_col_names = ["sent_id", "label"]
        self.output_file_path = output_file_path
        self.is_test = is_test

    def read(self):

        sent_df = pd.read_csv(self.sent_path, sep="\t", header=None, encoding="utf-8", names=self.sent_col_names)

        sent_relation_df = pd.read_csv(self.sent_relation_path, sep="\t", header=None, encoding="utf-8",
                                       names=self.sent_relation_col_names)

        sent_relation_df["label"] = sent_relation_df["label"].apply(str).apply(lambda label: label.split(" ")[0])

        return sent_df, sent_relation_df

    def get_all_words(self, series):

        all_words = []
        series.apply(lambda words: all_words.extend(words))

        return all_words

    def gen_unique_labels(self, labels):

        unique_labels = sorted(list(set(labels)))

        return unique_labels

    def padding(self, sequence, pad):

        sequence_length = len(sequence)
        if sequence_length > self.max_sequence_length:
            new_sequence = sequence[: self.max_sequence_length]
        else:
            padding_words = [pad] * (self.max_sequence_length - sequence_length)
            new_sequence = sequence[:]
            new_sequence.extend(padding_words)

        return new_sequence

    def next_batch(self, x, y, batch_size):
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

    def get_output_data(self):

        sent_df, sent_relation_df = self.read()

        # tmp = sent_relation_df["label"].value_counts()

        if self.is_test:
            df = sent_df

        df = pd.merge(left=sent_df, right=sent_relation_df, how="left", on="sent_id")

        # tmp_1 = df["label"].value_counts()

        df["sent"] = df["sent"].apply(str)
        df["label"] = df["label"].apply(int)

        df["sent"] = "<GO> " + df["per_1"] + " <SPACING> " + df["per_2"] + " <SPACING> " + df["sent"] + " <EOS>"
        df["sent"] = df["sent"].apply(str.split)

        finally_df = df[["sent_id", "sent", "label"]]

        return finally_df