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

import os

import numpy as np
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
from numpy import *

from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

class MovieReviewDataset(Dataset):
    """
    영화리뷰 데이터를 읽어서, tuple (데이터, 레이블)의 형태로 리턴하는 파이썬 오브젝트 입니다.
    """

    def __init__(self, dataset_path, max_length, is_train =True):
        """
        initializer

        :param dataset_path: 데이터셋 root path
        :param max_length: 문자열의 최대 길이
        """
        # 데이터, 레이블 각각의 경로
        if is_train == True:
            data_review = os.path.join(dataset_path, 'train', 'train_data')
            data_label = os.path.join(dataset_path, 'train', 'train_label')
        else:
            data_review = os.path.join(dataset_path, 'test', 'test_data')
            data_label = os.path.join(dataset_path, 'test', 'test_label')
        
        # 영화리뷰 데이터를 읽고 preprocess까지 진행합니다
        with open(data_review, 'rt') as f:
            text = []
            text_dataset = f.readlines()
            for t in text_dataset: 
                text.append(t)
                #text.append(t[::-1])
            self.reviews, self.depth = preprocess(text, max_length)

        # 영화리뷰 레이블을 읽고 preprocess까지 진행합니다.
        with open(data_label) as f:
            labels = []
            labels_dataset = f.readlines()

            for l in labels_dataset:
                labels.append(l)
                #labels.append(l)
            self.labels = [np.float32(x) for x in labels]

    def __len__(self):
        """

        :return: 전체 데이터의 수를 리턴합니다
        """
        return len(self.reviews)

    def __getitem__(self, idx):
        """

        :param idx: 필요한 데이터의 인덱스
        :return: 인덱스에 맞는 데이터, 레이블 pair를 리턴합니다
        """
        return self.reviews[idx], self.labels[idx]

    def get_depth(self):
        return self.depth

def preprocess(data, max_length):
    """
     입력을 받아서 딥러닝 모델이 학습 가능한 포맷으로 변경하는 함수입니다.
     기본 제공 알고리즘은 char2vec이며, 기본 모델이 MLP이기 때문에, 입력 값의 크기를 모두 고정한 벡터를 리턴합니다.
     문자열의 길이가 고정값보다 길면 긴 부분을 제거하고, 짧으면 0으로 채웁니다.

    :param data: 문자열 리스트 ([문자열1, 문자열2, ...])
    :param max_length: 문자열의 최대 길이
    :return: 벡터 리스트 ([[0, 1, 5, 6], [5, 4, 10, 200], ...]) max_length가 4일 때
    """
    
    tokenized_data = [list(row) for row in data]
    #for row in data:
    #    print("KK")
    #    tokenized_data.append(list(row))
    #    tokenized_data.append(list(row[::-1]))

    tokens = [t for d in tokenized_data for t in d[0]] # contain included all syllable

    max_length_Syllable = []
    for idx, seq in enumerate(tokenized_data):
        num_size = max_length - len(seq)
        for i in range(num_size):
            seq = np.append(seq, '')
        max_length_Syllable = np.append(max_length_Syllable, seq)
    max_length_Syllable = np.reshape(max_length_Syllable, (-1, max_length))
    #print(max_length_Syllable[0])

    print(max_length_Syllable.shape)
    max_length_Syllable, depth = kor_transform(max_length_Syllable, max_length)
    
    return max_length_Syllable, depth

def kor_transform(data, max_length):
    tokens = [t for d in data for t in d] # contain included all syllable
    label_encoder = preprocessing.LabelEncoder()
    label_encoder = label_encoder.fit(tokens)
    encoded_data = []
    depth = 0
    for line in data:
        encoded_line = np.array(label_encoder.transform(line),dtype=np.int32)
        encoded_data.append(np.array(label_encoder.transform(line),dtype=np.int32))
        depth = encoded_line.max() + 1
    data = None
    return encoded_data, depth

