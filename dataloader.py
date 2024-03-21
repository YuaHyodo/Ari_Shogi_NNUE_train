"""
python-dlshogi2のデータローダー
( https://github.com/TadaoYamaoka/python-dlshogi2/blob/main/pydlshogi2/dataloader.py )をベースにしている
"""
import os
import gc
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch

from cshogi import Board, HuffmanCodedPosAndEval, BLACK, WHITE, BLACK_WIN, WHITE_WIN, move_to_usi
from features_halfkp import *

def make_result(game_result, color):
    if color == BLACK:
        if game_result == BLACK_WIN:
            return 1
        if game_result == WHITE_WIN:
            return 0
    else:
        if game_result == BLACK_WIN:
            return 0
        if game_result == WHITE_WIN:
            return 1
    return 0.5

def dire_list_to_files(dire_list):
    if type(dire_list) == str:
        dire_list = [dire_list]
    files = []
    for dire in dire_list:
        f = os.listdir(dire)
        for i in f:
            if '.hcpe' not in i:
                continue
            files.append(dire + i)
    np.random.shuffle(files)
    return files

class HcpeDataLoader:
    def __init__(self, files, batch_size, device, shuffle=False, eval_a=600, pin_memory=True, unique=True):
        self.unique = unique
        
        #評価値を勝率に変換する際に使う値。データから調整したほうが良い。
        #https://tadaoyamaoka.hatenablog.com/entry/2021/05/06/213506
        self.eval_a = eval_a
        if self.eval_a > 0:
            self.use_eval = True
        else:
            self.use_eval = False
            
        self.load(files)
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle

        self.X1_torch = torch.empty((self.batch_size, features_num), dtype=torch.float32, pin_memory=pin_memory)
        self.X2_torch = torch.empty((self.batch_size, features_num), dtype=torch.float32, pin_memory=pin_memory)
        self.result_torch = torch.empty((batch_size, 1), dtype=torch.float32, pin_memory=pin_memory)
        self.value_torch = torch.empty((batch_size, 1), dtype=torch.float32, pin_memory=pin_memory)
        self.X1 = self.X1_torch.numpy()
        self.X2 = self.X2_torch.numpy()
        self.result = self.result_torch.numpy().reshape(-1)
        self.value = self.value_torch.numpy().reshape(-1)
        
        self.i = 0
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.board = Board()

    def load(self, files):
        self.data = []
        for path in files:
            self.data.append(np.fromfile(path, dtype=HuffmanCodedPosAndEval))
        self.data = np.concatenate(self.data)
        if self.unique:
            self.data = np.unique(self.data, axis=0)
        if self.use_eval:
            for i in range(len(self.data)):
                if self.data[i]['eval'] == 0:
                    continue
                score = self.data[i]['eval']
                self.data[i]['eval'] = (1.0 / (1.0 + np.exp(-score / self.eval_a))) * 1000

    def hcpe_to_value_output(self, hcpe):
        if not self.use_eval or hcpe['eval'] == 0:
            return make_result(hcpe['gameResult'], self.board.turn)
        return hcpe['eval'] / 1000

    def mini_batch(self, hcpevec):
        self.X1.fill(0)
        self.X2.fill(0)
        for i, hcpe in enumerate(hcpevec):
            self.board.set_hcp(hcpe['hcp'])
            make_input_features(self.board, self.X1[i], self.X2[i], fill_zero=False)
            self.result[i] = make_result(hcpe['gameResult'], self.board.turn)
            self.value[i] = self.hcpe_to_value_output(hcpe)

        if self.device.type == 'cpu':
            return (self.X1_torch.clone(), self.X2_torch.clone(), 
                    self.result_torch.clone(), self.value_torch.clone(),)
        else:
            return (self.X1_torch.to(self.device), self.X2_torch.to(self.device), 
                    self.result_torch.to(self.device), self.value_torch.to(self.device))

    def sample(self):
        return self.mini_batch(np.random.choice(self.data, self.batch_size, replace=False))

    def pre_fetch(self):
        hcpevec = self.data[self.i:self.i+self.batch_size]
        self.i += self.batch_size
        if len(hcpevec) < self.batch_size:
            return

        self.f = self.executor.submit(self.mini_batch, hcpevec)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        self.i = 0
        if self.shuffle:
            np.random.shuffle(self.data)
        self.pre_fetch()
        return self

    def __next__(self):
        if self.i > len(self.data):
            raise StopIteration()

        result = self.f.result()
        self.pre_fetch()

        return result
