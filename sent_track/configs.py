#! /usr/bin/env python
# coding: utf-8


"""

author: LCG22
date: 2020-02-22

content: 配置类文件

"""


class BaseConfig(object):

    def __init__(self):
        self.batch_size = 128


class ProcessingDataConfig(BaseConfig):

    def __init__(self):
        super(ProcessingDataConfig, self).__init__()

        self.open_data_path = "../open_data/"
        self.sent_train_path = self.open_data_path + "sent_train.txt"
        self.sent_dev_path = self.open_data_path + "sent_dev.txt"
        self.sent_test_path = self.open_data_path + "sent_test.txt"
        self.sent_relation_train_path = self.open_data_path + "sent_relation_train.txt"
        self.sent_relation_dev_path = self.open_data_path + "sent_relation_dev.txt"
        self.sent_relation_test_path = self.open_data_path + "sent_relation_test.txt"
        self.relation2id_path = self.open_data_path + "relation2id.txt"

        self.processed_sent_train_path = self.open_data_path + "processed_sent_train.csv"
        self.processed_sent_dev_path = self.open_data_path + "processed_sent_dev.csv"
        self.processed_sent_test_path = self.open_data_path + "processed_sent_test.csv"

        self.max_sequence_length = 50


class TrainingConfig(BaseConfig):

    def __init__(self):
        super(TrainingConfig, self).__init__()

        self.learning_rate = 1 * 10 ** -3
        self.epoches = 1
        # 每 n 步评估一下模型学习效果
        self.evaluate_every = 1000
        # 每 n 步保存一次模型
        self.checkpoint_every = 1000
        # 保存模型的目录
        self.save_model_path = "../save_model/"


class ModelConfig(BaseConfig):

    def __init__(self):
        super(ModelConfig, self).__init__()

        self.embedding_size = 300
        self.num_classes = 35
        self.dropout_keep_prob = 0.5
        # 词的数量为此值，若实际值比该值小则以实际值为准
        self.size_of_word = 1 * 10 ** 6
        # 隐藏层的数量和各层的神经元的个数
        self.hidden_sizes = [128, 64, 128]
        # 惩罚项 l2 的系数
        self.l2_reg_lambda = 0.0
        # 序列的最大长度
        self.max_sequence_length = ProcessingDataConfig().max_sequence_length


class RunModelConfig(BaseConfig):

    def __init__(self):
        super(RunModelConfig, self).__init__()

        self.training_config = TrainingConfig()
        self.model_config = ModelConfig()

