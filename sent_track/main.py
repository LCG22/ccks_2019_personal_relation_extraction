#! /usr/bin/env python
# coding: utf-8


"""

author: LCG22
date: 2020-02-22

content:
1、处理数据
2、训练模型

"""

try:
    from .configs import ProcessingDataConfig, RunModelConfig
except Exception as e:
    from configs import ProcessingDataConfig, RunModelConfig

try:
    from .processing_raw_data import ProcessingRawData
except Exception as e:
    from processing_raw_data import ProcessingRawData

try:
    from .create_vocab import Vocab
except Exception as e:
    from create_vocab import Vocab

try:
    from .estimate import multi_f_beta, multi_precision, multi_recall, get_multi_metrics, mean
except Exception as e:
    from estimate import multi_f_beta, multi_precision, multi_recall, get_multi_metrics, mean

try:
    from .bi_lstm_attention_model import BiLSTM
except Exception as e:
    from bi_lstm_attention_model import BiLSTM

import numpy as np
import tensorflow as tf
import datetime
import os
import pandas as pd

if __name__ == '__main__':

    pro_conf = ProcessingDataConfig()
    run_model_config = RunModelConfig()

    train_pro_raw_data = ProcessingRawData(sent_path=pro_conf.sent_train_path,
                                           sent_relation_path=pro_conf.sent_relation_train_path,
                                           output_file_path=pro_conf.processed_sent_train_path,
                                           max_sequence_length=pro_conf.max_sequence_length)
    train_data = train_pro_raw_data.get_output_data()

    # 评估模型数据集
    dev_pro_raw_data = ProcessingRawData(sent_path=pro_conf.sent_dev_path,
                                         sent_relation_path=pro_conf.sent_relation_dev_path,
                                         output_file_path=pro_conf.processed_sent_dev_path,
                                         max_sequence_length=pro_conf.max_sequence_length)
    dev_data = dev_pro_raw_data.get_output_data()

    # 打乱数据集
    train_data = train_data.sample(train_data.shape[0])
    train_data_0 = train_data[train_data["label"].astype(int) == 0]
    train_data_1 = train_data[train_data["label"].astype(int) != 0]
    num = int(train_data_0.shape[0] / (train_data_1.shape[0]))
    train_data_1 = pd.concat([train_data_1] * num, axis=0)
    train_data = pd.concat([train_data_0, train_data_1], axis=0)
    train_data = train_data.reset_index(drop=True)
    # 打乱数据集
    train_data = train_data.sample(train_data.shape[0])
    # # 打乱数据集
    # train_data = train_data.sample(train_data.shape[0])
    # train_data_0 = train_data[train_data["label"].astype(int) == 0]
    # train_data_1 = train_data[train_data["label"].astype(int) != 0]
    # train_data_0 = train_data_0[: train_data_1.shape[0]]
    # train_data = pd.concat([train_data_0, train_data_1], axis=0)
    # train_data = train_data.reset_index(drop=True)
    # # 打乱数据集
    # train_data = train_data.sample(train_data.shape[0])
    # 打乱数据集
    # train_data = train_data.sample(train_data.shape[0])

    all_words = train_pro_raw_data.get_all_words(series=train_data["sent"])
    # 将测试集中的人名加入到词典中
    dev_data["sent"].apply(lambda sequence: all_words.extend(sequence[1: 4: 2]))

    vocab = Vocab()
    vocab.build_vocab(words=all_words, size=run_model_config.model_config.size_of_word)

    sent_length_describe = train_data["sent"].apply(len).describe()
    print("sent length describe: \n{}".format(sent_length_describe))
    """
    count    281241.000000
    mean         26.761624
    std         111.459805
    min           8.000000
    25%          20.000000
    50%          26.000000
    75%          33.000000
    max       41266.000000

    """

    train_data["sent"] = train_data["sent"].apply(lambda sentence: train_pro_raw_data.padding(sequence=sentence,
                                                                                              pad=vocab.get_pad()))
    train_data["sent"] = train_data["sent"].apply(lambda sentence: vocab.get_words_to_index(sentence))

    unique_labels = train_pro_raw_data.gen_unique_labels(train_data["label"].tolist())
    num_label = len(unique_labels)

    # 打乱数据集
    # dev_data = dev_data.sample(dev_data.shape[0])

    sent_length_describe = dev_data["sent"].apply(len).describe()
    print("dev sent length describe: \n{}".format(sent_length_describe))
    """
    count    37637.000000
    mean        26.744400
    std         67.149995
    min          8.000000
    25%         20.000000
    50%         26.000000
    75%         33.000000
    max       9617.000000
    """

    dev_data["sent"] = dev_data["sent"].apply(lambda sentence: dev_pro_raw_data.padding(sequence=sentence,
                                                                                        pad=vocab.get_pad()))
    dev_data["sent"] = dev_data["sent"].apply(lambda sentence: vocab.get_words_to_index(sentence))

    # 生成训练集和验证集
    train_x = np.array(train_data["sent"].tolist())
    train_y = np.array(train_data["label"].tolist())
    eval_x = np.array(dev_data["sent"].tolist())
    eval_y = np.array(dev_data["label"].tolist())

    run_model_config.model_config.size_of_word = run_model_config.model_config.size_of_word \
        if run_model_config.model_config.size_of_word < len(vocab.word2idx) else len(vocab.word2idx)

    word_embedding = np.random.random([len(vocab.word2idx), run_model_config.model_config.embedding_size])
    unique_labels = unique_labels

    # 定义计算图
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_conf.gpu_options.allow_growth = True
        session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9  # 配置gpu占用率

        sess = tf.Session(config=session_conf)

        # 定义会话
        with sess.as_default():
            lstm = BiLSTM(run_model_config.model_config, word_embedding)
            lstm.gen_model()

            global_step = tf.Variable(0, name="global_step", trainable=False)
            # 定义优化函数，传入学习速率参数
            optimizer = tf.train.AdamOptimizer(run_model_config.training_config.learning_rate)
            # 计算梯度,得到梯度和变量
            grads_and_vars = optimizer.compute_gradients(lstm.loss)
            # 将梯度应用到变量下，生成训练器
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # 用summary绘制tensorBoard
            # gradSummaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    try:
                        tf.summary.histogram("{}/grad/hist".format(v.name), g)
                        tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    except Exception as e:
                        print(e)

            out_dir = os.path.abspath(os.path.join(os.path.curdir, "summarys"))
            print("Writing to {}\n".format(out_dir))

            loss_summary = tf.summary.scalar("loss", lstm.loss)
            try:
                summary_op = tf.summary.merge_all()
            except Exception as e:
                print(e)

            train_summary_dir = os.path.join(out_dir, "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            eval_summary_dir = os.path.join(out_dir, "eval")
            eval_summary_writer = tf.summary.FileWriter(eval_summary_dir, sess.graph)

            # 初始化所有变量
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

            # 保存模型的一种方式，保存为pb文件
            save_model_path = run_model_config.training_config.save_model_path
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
                    lstm.dropout_keep_prob: run_model_config.model_config.dropout_keep_prob
                }
                # _, summary, step, loss, predictions = sess.run(
                #     [train_op, summary_op, global_step, lstm.loss, lstm.predictions],
                #     feed_dict)
                _, step, loss, predictions = sess.run(
                    [train_op, global_step, lstm.loss, lstm.predictions],
                    feed_dict)

                if run_model_config.model_config.num_classes == 2:
                    acc, recall, prec, f_beta = get_binary_metrics(pred_y=predictions, true_y=batch_y)

                elif run_model_config.model_config.num_classes > 2:
                    acc, recall, prec, f_beta = get_multi_metrics(pred_y=predictions, true_y=batch_y,
                                                                  labels=unique_labels)

                # try:
                #     train_summary_writer.add_summary(summary, step)
                # except Exception as e:
                #     print(e)

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
                # summary, step, loss, predictions = sess.run(
                #     [summary_op, global_step, lstm.loss, lstm.predictions],
                #     feed_dict)
                step, loss, predictions = sess.run(
                    [global_step, lstm.loss, lstm.predictions],
                    feed_dict)

                if run_model_config.model_config.num_classes == 2:

                    acc, precision, recall, f_beta = get_binary_metrics(pred_y=predictions, true_y=batch_y)
                elif run_model_config.model_config.num_classes > 2:
                    acc, precision, recall, f_beta = get_multi_metrics(pred_y=predictions, true_y=batch_y,
                                                                       labels=unique_labels)

                # eval_summary_writer.add_summary(summary, step)

                return loss, acc, precision, recall, f_beta


            for i in range(run_model_config.training_config.epoches):
                # 训练模型
                print("start training_config model")
                for batch_x, batch_y in train_pro_raw_data.next_batch(train_x, train_y, run_model_config.batch_size):
                    loss, acc, prec, recall, f_beta = train_step(batch_x, batch_y)

                    current_step = tf.train.global_step(sess, global_step)
                    time_str = datetime.datetime.now().isoformat()
                    print(
                        "current time: {}, train: step: {}, loss: {}, acc: {}, recall: {}, precision: {}, f_beta: {}".format(
                            time_str, current_step, loss, acc, recall, prec, f_beta))
                    if current_step % run_model_config.training_config.evaluate_every == 0:
                        print("\nEvaluation:")

                        losses = []
                        accs = []
                        f_betas = []
                        precisions = []
                        recalls = []

                        for eval_batch_x, eval_batch_y in dev_pro_raw_data.next_batch(eval_x, eval_y,
                                                                                      run_model_config.batch_size):
                            loss, acc, precision, recall, f_beta = dev_step(eval_batch_x, eval_batch_y)
                            losses.append(loss)
                            accs.append(acc)
                            f_betas.append(f_beta)
                            precisions.append(precision)
                            recalls.append(recall)
                            1

                        time_str = datetime.datetime.now().isoformat()
                        print(
                            "current time: {}, step: {}, loss: {}, acc: {},precision: {}, recall: {}, f_beta: {}".format(
                                time_str,
                                current_step,
                                mean(losses),
                                mean(accs),
                                mean(precisions),
                                mean(recalls),
                                mean(f_betas)))

                    if current_step % run_model_config.training_config.checkpoint_every == 0:
                        # 保存模型的另一种方法，保存checkpoint文件
                        path = saver.save(sess, "./save_model/", global_step=current_step)
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

1
