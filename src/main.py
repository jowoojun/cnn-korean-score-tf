# -*- coding: utf-8 -*-

"""
Copyright 2018 NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import argparse
import os

import numpy as np
import tensorflow as tf

import nsml
from nsml import DATASET_PATH, HAS_DATASET, GPU_NUM, IS_ON_NSML
from dataset import MovieReviewDataset, preprocess

from sklearn.model_selection import train_test_split

# DONOTCHANGE: They are reserved for nsml
# This is for nsml leaderboard
def bind_model(sess, config, model):
    # 학습한 모델을 저장하는 함수입니다.
    def save(dir_name, *args):
        # directory
        os.makedirs(dir_name, exist_ok=True)
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(dir_name, 'model'))

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(dir_name, *args):
        saver = tf.train.Saver()
        # find checkpoint
        ckpt = tf.train.get_checkpoint_state(dir_name)
        if ckpt and ckpt.model_checkpoint_path:
            checkpoint = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(dir_name, checkpoint))
        else:
            raise NotImplemented('No checkpoint!')
        print('Model loaded')

    def infer(raw_data, **kwargs):
        """

        :param raw_data: raw input (여기서는 문자열)을 입력받습니다
        :param kwargs:
        :return:
        """
        # dataset.py에서 작성한 preprocess 함수를 호출하여, 문자열을 벡터로 변환합니다
        preprocessed_data = preprocess(raw_data, config.strmaxlen)
        # 저장한 모델에 입력값을 넣고 prediction 결과를 리턴받습니다
        #pred = sess.run(output_sigmoid, feed_dict={x: preprocessed_data})
        pred = model[0].predict(preprocessed_data[0])
        clipped = np.array(pred > config.threshold, dtype=np.int)
        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(확률, 0 or 1)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 리더보드 결과에 확률의 값은 영향을 미치지 않습니다
        return list(zip(pred.flatten(), clipped.flatten()))

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)


def _batch_loader(iterable, n=1):
    """
    데이터를 배치 사이즈만큼 잘라서 보내주는 함수입니다. PyTorch의 DataLoader와 같은 역할을 합니다

    :param iterable: 데이터 list, 혹은 다른 포맷
    :param n: 배치 사이즈
    :return:
    """
    length = len(iterable)
    for n_idx in range(0, length, n):
        yield iterable[n_idx:min(n_idx + n, length)]


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

class Model:
    def __init__(self, sess, name, config):
        self.sess = sess
        self.name = name
        self._build_net(config)
        
    def _build_net(self, config):
        self.config = config
        self.output_size = 11
        self.hidden_layer_size = 200
        self.learning_rate = 0.01
        self.input_size = self.config.embedding * self.config.strmaxlen
        self.character_size = 11172
        filter_sizes = list(map(int, self.config.filter_sizes.split(",")))


        with tf.variable_scope(self.name):
            # Placeholders for input, output and dropout
            self.input_x = tf.placeholder(tf.int32, [None, self.config.strmaxlen], name="input_x")
            self.input_y = tf.placeholder(tf.float32, [None, self.output_size], name="input_y")
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

            # Keeping track of l2 regularization loss (optional)
            l2_loss = tf.constant(0.0)

            # Embedding layer
            with tf.device('/cpu:0'), tf.name_scope("embedding"):
                self.W = tf.Variable(
                    tf.random_uniform([self.character_size, self.config.embedding], -1.0, 1.0),
                    name="W")
                self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
                self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, self.config.embedding, 1, self.config.num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[self.config.num_filters]), name="b")
                    conv = tf.nn.conv2d(
                        self.embedded_chars_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, self.config.strmaxlen - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)

            # Combine all the pooled features
            num_filters_total = self.config.num_filters * len(filter_sizes)
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

            # Add dropout
            with tf.name_scope("dropout"):
                self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

            # Final (unnormalized) scores and predictions
            with tf.name_scope("output"):
                W = tf.get_variable(
                    "W",
                    shape=[num_filters_total, self.output_size],
                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[self.output_size]), name="b")
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                self.scores = tf.matmul(self.h_drop, W) + b
                self.output_sigmoid = tf.sigmoid(self.scores) * 9 + 1
                self.predictions = tf.argmax(self.scores, 1, name="predictions")

            # Calculate mean cross-entropy loss
            with tf.name_scope("loss"):
                #losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
                #losses = tf.losses.mean_squared_error(self.input_y, self.output_sigmoid)
                self.diff = self.input_y - self.output_sigmoid
                self.losses = tf.reduce_mean(tf.square(self.diff))
                #-(self.input_y * tf.log(self.output_sigmoid)) - (1-self.input_y) * tf.log(1-self.output_sigmoid))
                self.binary_cross_entropy = self.losses + self.config.l2_reg_lambda * l2_loss
                self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.binary_cross_entropy)

            # Accuracy
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def predict(self, x_test, keep_prop=1.0):
        return self.sess.run(self.output_sigmoid, 
                             feed_dict={self.input_x:x_test, self.dropout_keep_prob: keep_prop})

    def get_accuracy(self, x_test, y_test, keep_prop=1.0): 
        return self.sess.run(self.accuracy, 
                             feed_dict={self.input_x: x_test, self.input_y: y_test, self.dropout_keep_prob: keep_prop})

    def train(self, x_data, y_data, keep_prop=0.5):
        return self.sess.run([self.train_step, self.binary_cross_entropy], 
                        feed_dict={self.input_x: x_data, self.input_y: y_data, self.dropout_keep_prob:keep_prop})

    def get_diff(self, x_data, y_data, keep_prop=0.5):
        print(np.reshape(self.sess.run([self.diff],
                            feed_dict={self.input_x: x_data, self.input_y: y_data, self.dropout_keep_prob:keep_prop}), (-1)))

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')

    # User options
    args.add_argument('--output', type=int, default=1)
    args.add_argument('--epochs', type=int, default=10)
    args.add_argument('--batch', type=int, default=2000)
    args.add_argument('--strmaxlen', type=int, default=200)
    args.add_argument('--embedding', type=int, default=16)
    args.add_argument('--threshold', type=float, default=0.5)
    args.add_argument('--filter_sizes', type=str, default="3,4")
    args.add_argument('--num_filters', type=int, default=100)
    args.add_argument('--l2_reg_lambda', type=float, default=0.0)
    config = args.parse_args()

    if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
        # poem data
        #DATASET_PATH = '../sample_data/movie_review/'
        #TESTSET_PATH = '../sample_data/movie_review/'
        # fake data
        DATASET_PATH = '../fake_data/movie_review/'
        TESTSET_PATH = '../fake_data/movie_review/'

    sess = tf.InteractiveSession()
   
    models = []
    num_models = 3
    for m in range(num_models):
        models.append(Model(sess, "model"+str(m), config))
    
    tf.global_variables_initializer().run()

    # DONOTCHANGE: Reserved for nsml
    bind_model(sess=sess, config=config, model=models)

    # DONOTCHANGE: Reserved for nsml
    if config.pause:
        nsml.paused(scope=locals())

    if config.mode == 'train':
        # 데이터를 로드합니다.
        dataset = MovieReviewDataset(DATASET_PATH, config.strmaxlen, is_train=True)
        dataset_len = len(dataset)
        one_batch_size = dataset_len//config.batch
        if dataset_len % config.batch != 0:
            one_batch_size += 1

        
        # epoch마다 학습을 수행합니다.
        for epoch in range(config.epochs):
            #avg_loss = 0.0
            avg_cost_list = np.zeros(len(models))
            for i, (data, labels) in enumerate(_batch_loader(dataset, config.batch)):
                labels = np.reshape(labels,(-1, 1))
                onehot_label = sess.run(tf.reshape(tf.one_hot(labels, depth=11, dtype=tf.float32), (-1,11)))
                
                for m_idx, m in enumerate(models):
                    #m.get_diff(data, labels)
                    _, loss = m.train(data, onehot_label)
                    print('Batch : ', i + 1, '/', one_batch_size,
                          ', BCE in this minibatch: ', float(loss))
                    avg_cost_list[m_idx] += float(loss) / one_batch_size

            print('epoch:', epoch, ' train_loss:', avg_cost_list)
            #nsml.report(summary=True, scope=locals(), epoch=epoch, epoch_total=config.epochs,
            #            train__loss=float(avg_loss/one_batch_size), step=epoch)
            nsml.report(summary=True, scope=locals(), epoch=epoch, epoch_total=config.epochs,
                        train__loss=avg_cost_list[0], step=epoch)
            # DONOTCHANGE (You can decide how often you want to save the model)
            nsml.save(epoch)
        
        test_dataset = MovieReviewDataset(TESTSET_PATH, config.strmaxlen, is_train=False)
        test_dataset_len = len(test_dataset)
        test_one_batch_size = test_dataset_len//config.batch
        if test_dataset_len % config.batch != 0:
            test_one_batch_size += 1

        for i, (data, labels) in enumerate(_batch_loader(test_dataset, config.batch)):
            labels = np.reshape(labels, (-1,1))
            onehot_label = sess.run(tf.reshape(tf.one_hot(labels, depth=11, dtype=tf.float32), (-1,11)))
            # Test model and check accuracy
            predictions = np.zeros(test_dataset_len * 11).reshape(test_dataset_len, 11)
            
            for m_idx, m in enumerate(models):
                print(m_idx, 'Accuracy:', m.get_accuracy(data, onehot_label)) 
                p = m.predict(data)
                predictions += p

            print(sess.run(tf.argmax(predictions, 1))
            print(sess.run(np.reshape(labels, (-1))))

            ensemble_correct_prediction = tf.equal(tf.argmax(predictions, 1), 
                                                   tf.argmax(labels, 1))              
            ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))
                
            print('Ensemble accuracy:', sess.run(ensemble_accuracy))
                        

    # 로컬 테스트 모드일때 사용합니다
    # 결과가 아래와 같이 나온다면, nsml submit을 통해서 제출할 수 있습니다.
    # [(0.3, 0), (0.7, 1), ... ]
    elif config.mode == 'test_local':
        with open(os.path.join(DATASET_PATH, 'train/train_data'), 'rt', encoding='utf-8') as f:
            queries = f.readlines()
        res = []
        for batch in _batch_loader(queries, config.batch):
            temp_res = nsml.infer(batch)
            res += temp_res
        print(res)
