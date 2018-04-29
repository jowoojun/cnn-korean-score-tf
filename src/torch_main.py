
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
import torch

from torch.autograd import Variable
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn.init as init

import nsml
from dataset import MovieReviewDataset, preprocess
from nsml import DATASET_PATH, HAS_DATASET, GPU_NUM, IS_ON_NSML
import nltk

from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

# DONOTCHANGE: They are reserved for nsml
# This is for nsml leaderboard
def bind_model(model, config):
    # 학습한 모델을 저장하는 함수입니다.
    def save(filename, *args):
        checkpoint = {
            'model': model.state_dict()
        }
        torch.save(checkpoint, filename)

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(filename, *args):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model'])
        print('Model loaded')

    def infer(raw_data, **kwargs):
        """
        :param raw_data: raw input (여기서는 문자열)을 입력받습니다
        :param kwargs:
        :return:
        """
        # dataset.py에서 작성한 preprocess 함수를 호출하여, 문자열을 벡터로 변환합니다
        preprocessed_data = preprocess(raw_data, config.strmaxlen)
        model.eval()
        # 저장한 모델에 입력값을 넣고 prediction 결과를 리턴받습니다
        output_prediction = model(preprocessed_data[0])
        point = output_prediction.data.squeeze(dim=1).tolist()
        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(confidence interval, 포인트)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 리더보드 결과에 confidence interval의 값은 영향을 미치지 않습니다
        return list(zip(np.zeros(len(point)), point))

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)


def collate_fn(data: list):
    """
    PyTorch DataLoader에서 사용하는 collate_fn 입니다.
    기본 collate_fn가 리스트를 flatten하기 때문에 벡터 입력에 대해서 사용이 불가능해, 직접 작성합니다.
    :param data: 데이터 리스트
    :return:
    """
    review = []
    label = []
    for datum in data:
        review.append(datum[0])
        label.append(datum[1])
    # 각각 데이터, 레이블을 리턴
    return review, np.array(label)


class Regression(nn.Module):
    def __init__(self, args):
        super(Regression, self).__init__()
        self.args = args

        V = args.embed_num
        D = args.embedding
        self.D = args.embedding
        C = args.class_num
        Ci = 1
        hidden_layer1 = 320
        Co = args.kernel_num
        Ks = args.kernel_sizes

        self.embed = nn.Embedding(V, D)

        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])

        #self.dropout = nn.Dropout(args.dropout)
        self.convs_bn = nn.BatchNorm2d(len(Ks)*Co)

        self.fc1 = nn.Linear(len(Ks)*Co, hidden_layer1)

        self.fc1_bn = nn.BatchNorm1d(hidden_layer1)

        self.fc2 = nn.Linear(hidden_layer1, Ci)

        #self.fc2_bn = nn.BatchNorm1d(hidden_layer2)

        #self.fc3 = nn.Linear(hidden_layer2, Ci)
        #init.xavier_uniform(self.fc3.weight, gain=np.sqrt(1))
        #init.constant(self.fc3.bias, 0.1)


    def forward(self, data):
        # 임베딩의 차원 변환을 위해 배치 사이즈를 구합니다.
        batch_size = len(data)
        # list로 받은 데이터를 torch Variable로 변환합니다.
        # 뉴럴네트워크를 지나 결과를 출력합니다.
        data_in_torch = Variable(torch.from_numpy(np.array(data)).long())
        
        # 만약 gpu를 사용중이라면, 데이터를 gpu 메모리로 보냅니다.
        if GPU_NUM:
            data_in_torch = data_in_torch.cuda()
        # 뉴럴네트워크를 지나 결과를 출력합니다.
        data = self.embed(data_in_torch)

        if self.args.static:
            data = Variable(data)

        data = data.unsqueeze(1)  # (N, Ci, W, D)

        data = [F.relu(conv(data)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        data = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in data]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(data, 1)

        #x = self.dropout(x)  # (N, len(Ks)*Co)

        #x = self.convs_bn(x)

        #x = self.fc1(x)
        x = F.relu(self.fc1_bn(self.fc1(x)))

        #x = F.relu(self.fc2_bn(self.fc2(x)))
        # 영화 리뷰가 1~10점이기 때문에, 스케일을 맞춰줍니다
        #logit = self.fc1(x)  # (N, C)
        #output = torch.sigmoid(logit) * 9 + 1

        output = torch.sigmoid(self.fc2(x)) * 9 + 1
        #output = torch.sigmoid(self.fc3(x)) * 9 + 1

        return output


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
    args.add_argument('--dropout', type=int, default=1)
    args.add_argument('--strmaxlen', type=int, default=200)
    args.add_argument('--embedding', type=int, default=16)
    args.add_argument('--kernel-num', type=int, default=100)
    args.add_argument('--kernel-sizes', type=str, default='3,5', help='comma-separated kernel size to use for convolution')
    args.add_argument('--static', default=False)
    config = args.parse_args()

    if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
        DATASET_PATH = '../sample_data/movie_review/'

    config.embed_num = 11172
    config.class_num = 1
    config.kernel_sizes = [int(k) for k in config.kernel_sizes.split(',')]

    model = Regression(config)
    #model.apply(weights_init)

    if GPU_NUM:
        model = model.cuda()

    # DONOTCHANGE: Reserved for nsml use
    bind_model(model, config)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # DONOTCHANGE: They are reserved for nsml
    if config.pause:
        nsml.paused(scope=locals())

    # 학습 모드일 때 사용합니다. (기본값)
    if config.mode == 'train':
        # 데이터를 로드합니다.
        dataset = MovieReviewDataset(DATASET_PATH, config.strmaxlen)
        train_loader = DataLoader(dataset=dataset,
                                  batch_size=config.batch,
                                  shuffle=True,
                                  collate_fn=collate_fn,
                                  num_workers=2)
        total_batch = len(train_loader)

        #epoch마다 학습을 수행합니다.
        for epoch in range(config.epochs):
            avg_loss = 0.0
            for i, (data, labels) in enumerate(train_loader):
                predictions = model(data)
                label_vars = Variable(torch.from_numpy(labels))

                if GPU_NUM:
                    label_vars = label_vars.cuda()
                for n in range(len(predictions)):
                    print(label_vars[n] - predictions[n])

                loss = criterion(predictions, label_vars)

                if GPU_NUM:
                    loss = loss.cuda()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print('Batch : ', i + 1, '/', total_batch,
                      ', MSE in this minibatch: ', loss.data[0])
                avg_loss += loss.data[0]
            print('epoch:', epoch, ' train_loss:', float(avg_loss/total_batch))
            # nsml ps, 혹은 웹 상의 텐서보드에 나타나는 값을 리포트하는 함수입니다.
            #
            nsml.report(summary=True, scope=locals(), epoch=epoch, epoch_total=config.epochs,
                        train__loss=float(avg_loss/total_batch), step=epoch)
            # DONOTCHANGE (You can decide how often you want to save the model)
            nsml.save(epoch)

    # 로컬 테스트 모드일때 사용합니다
    # 결과가 아래와 같이 나온다면, nsml submit을 통해서 제출할 수 있습니다.
    # [(0.0, 9.045), (0.0, 5.91), ... ]
    elif config.mode == 'test_local':
        with open(os.path.join(DATASET_PATH, 'train/train_data'), 'rt', encoding='utf-8') as f:
            reviews = f.readlines()
        res = nsml.infer(reviews)
        print(res)
