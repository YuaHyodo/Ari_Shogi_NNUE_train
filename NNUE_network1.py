"""
NNUE-pytorch将棋版を参考に作った
https://github.com/nodchip/nnue-pytorch/blob/shogi.2022-10-28.sgd/model.py
https://github.com/nodchip/nnue-pytorch/blob/shogi.2022-10-28.sgd/serialize.py

一部は
https://github.com/bleu48/shogi-eval/blob/master/evaltest_nnue.py
https://github.com/bleu48/shogi-eval/blob/master/sample4-5a.py
も参考にしている。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from struct import unpack_from
import struct

from features_halfkp import*

VERSION = 0x7AF32F16

class NNUE_halfkp(nn.Module):
    def __init__(self, L1=256, L2=32, L3=32):
        super(NNUE_halfkp, self).__init__()
        self.units = (L1, L2, L3)
        self.input_layer = nn.Linear(features_num, L1)
        self.l1 = nn.Linear(2 * L1, L2)
        self.l2 = nn.Linear(L2, L3)
        self.output_layer = nn.Linear(L3, 1)
        
    def forward(self, x1, x2):
        x1 = self.input_layer(x1)
        x2 = self.input_layer(x2)
        x = torch.cat([x1, x2], dim=1)
        x = torch.clamp(x, 0.0, 1.0)
        x = torch.clamp(self.l1(x), 0.0, 1.0)
        x = torch.clamp(self.l2(x), 0.0, 1.0)
        x = self.output_layer(x)
        return x

def ascii_hist(name, x, bins=6):
    N,X = np.histogram(x, bins=bins)
    total = 1.0 * len(x)
    width = 50
    nmax = N.max()
    print(name)
    for (xi, n) in zip(X, N):
        bar = '#' * int(n * 1.0 * width / nmax)
        xi = '{0: <8.4g}'.format(xi).ljust(10)
        print('{0}| {1}'.format(xi, bar))

NUM_SQ = 81
NUM_PLANES = 1548

class NNUEWriter:
    def __init__(self, model):
        self.halfkp_hash = 0x5d69d5b8
        self.model = model
        self.buf = bytearray()
        fc_hash = self.fc_hash(self.model)
        self.write_header(model, fc_hash)
        
        self.int32(self.halfkp_hash ^ (model.units[0] * 2))
        self.write_feature_transformer(model)

        self.int32(fc_hash)
        self.write_fc_layer(model.l1)
        self.write_fc_layer(model.l2)
        self.write_fc_layer(model.output_layer, is_output=True)

    def fc_hash(self, model):
        prev_hash = 0xEC42E90D
        prev_hash ^= (model.units[0] * 2)
        layers = [model.l1, model.l2, model.output_layer]
        for layer in layers:
            layer_hash = 0xCC03DAE4
            layer_hash += layer.out_features
            layer_hash ^= prev_hash >> 1
            layer_hash ^= (prev_hash << 31) & 0xFFFFFFFF
            if layer.out_features != 1:
                layer_hash = (layer_hash + 0x538D24C7) & 0xFFFFFFFF
            prev_hash = layer_hash
        return layer_hash

    def write_header(self, model, fc_hash):
        self.int32(VERSION)
        #0x5d69d5b8 <= HalfKPのHash
        self.int32(fc_hash ^ self.halfkp_hash ^ (model.units[0] * 2))
        description = b"Features=HalfKP(Friend)[125388->256x2],"
        description += b"Network=AffineTransform[1<-32](ClippedReLU[32](AffineTransform[32<-32]"
        description += b"(ClippedReLU[32](AffineTransform[32<-512](InputSlice[512(0:512)])))))"
        self.int32(len(description)) # Network definition
        self.buf.extend(description)

    def get_virtual_to_real_features_gather_indices(self):
        factors = OrderedDict([('HalfKP', NUM_PLANES * NUM_SQ)])
        num_features = sum(v for n, v in factors.items())
        num_real_features = factors['HalfKP']
        num_virtual_features = num_features - num_real_features

        def get_factor_base_feature(name, f):
            offset = 0
            for n, s in f.items():
                if n == name:
                    return offset
                offset += s
            raise ValueError()
        
        #
        #
        indices = []
        real_offset = 0
        offset = 0
        for i_real in range(num_real_features):
            i_fact = [i_real]#HalfKP
            indices.append([offset + i for i in i_fact])
            
        real_offset += num_real_features
        offset += num_features
        return indices

    def coalesce_ft_weights(self, model, layer):
        weight = layer.weight.data
        indices = self.get_virtual_to_real_features_gather_indices()
        weight_coalesced = weight.new_zeros((weight.shape[0], NUM_PLANES * NUM_SQ))
        for i_real, is_virtual in enumerate(indices):
            weight_coalesced[:, i_real] = sum(weight[:, i_virtual] for i_virtual in is_virtual)
        return weight_coalesced

    def write_feature_transformer(self, model):
        # int16 bias = round(x * 127)
        # int16 weight = round(x * 127)
        layer = model.input_layer
        bias = layer.bias.data
        bias = bias.mul(127).round().to(torch.int16)
        ascii_hist('ft bias:', bias.numpy())
        self.buf.extend(bias.flatten().numpy().tobytes())
        print(np.array(layer.weight.data).shape)
        weight = self.coalesce_ft_weights(model, layer)
        weight = weight.mul(127).round().to(torch.int16)
        ascii_hist('ft weight:', weight.numpy())
        # weights stored as [41024][256], so we need to transpose the pytorch [256][41024]
        self.buf.extend(weight.transpose(0, 1).flatten().numpy().tobytes())

    def write_fc_layer(self, layer, is_output=False):
        # FC layers are stored as int8 weights, and int32 biases
        kWeightScaleBits = 6
        kActivationScale = 127.0
        if not is_output:
            kBiasScale = (1 << kWeightScaleBits) * kActivationScale # = 8128
        else:
            kBiasScale = 9600.0 # kPonanzaConstant * FV_SCALE = 600 * 16 = 9600
        kWeightScale = kBiasScale / kActivationScale # = 64.0 for normal layers
        kMaxWeight = 127.0 / kWeightScale # roughly 2.0

        # int32 bias = round(x * kBiasScale)
        # int8 weight = round(x * kWeightScale)
        bias = layer.bias.data
        bias = bias.mul(kBiasScale).round().to(torch.int32)
        ascii_hist('fc bias:', bias.numpy())
        self.buf.extend(bias.flatten().numpy().tobytes())
        weight = layer.weight.data
        clipped = torch.count_nonzero(weight.clamp(-kMaxWeight, kMaxWeight) - weight)
        total_elements = torch.numel(weight)
        clipped_max = torch.max(torch.abs(weight.clamp(-kMaxWeight, kMaxWeight) - weight))
        print("layer has {}/{} clipped weights. Exceeding by {} the maximum {}.".format(clipped, total_elements, clipped_max, kMaxWeight))
        weight = weight.clamp(-kMaxWeight, kMaxWeight).mul(kWeightScale).round().to(torch.int8)
        ascii_hist('fc weight:', weight.numpy())
        # FC inputs are padded to 32 elements for simd.
        num_input = weight.shape[1]
        if num_input % 32 != 0:
            num_input += 32 - (num_input % 32)
            new_w = torch.zeros(weight.shape[0], num_input, dtype=torch.int8)
            new_w[:, :weight.shape[1]] = weight
            weight = new_w
        # Stored as [outputs][inputs], so we can flatten
        self.buf.extend(weight.flatten().numpy().tobytes())

    def int32(self, v):
        self.buf.extend(struct.pack("<I", v))

    def write_to_file(self, file):
        with open(file, 'wb') as f:
            f.write(self.buf)
        return

class NNUEReader:
    def __init__(self, f):
        self.f = f
        self.model = NNUE_halfkp()
        self.read_weights()

    def read_weights(self):
        i = 178
        nn_data = self.f.read()
        input_layer_weight = np.array(unpack_from("<"+str(256*125388)+"h", nn_data, 16+i+256*2) )#.reshape(125388,256).T
        input_layer_bias = np.array(unpack_from("<"+str(256)+"h", nn_data, 16+i))
        weight_L1 = np.array(unpack_from("<"+str(512*32)+"b", nn_data, 16+i+2*(256+256*125388)+4 + 4*32))#.reshape(32,512)
        weight_L2 = np.array(unpack_from("<"+str(32*32)+"b", nn_data, 16+i+2*(256+256*125388)+4+ 4*32+32*512 + 4*32))#.reshape(32,32)
        weight_L3 = np.array(unpack_from("<"+str(32*1)+"b", nn_data, 16+i+2*(256+256*125388)+4+ 4*32+32*512 + 4*32+32*32 + 4*1))
        bias_L1 = np.array(unpack_from("<"+str(32)+"i", nn_data, 16+i+2*(256+256*125388)+4))
        bias_L2 = np.array(unpack_from("<"+str(32)+"i", nn_data, 16+i+2*(256+256*125388)+4+ 4*32+32*512))
        bias_L3 = np.array(unpack_from("<"+str(1)+"i", nn_data, 16+i+2*(256+256*125388)+4+ 4*32+32*512 + 4*32+32*32))

        

        self.model.input_layer.weight.data = torch.from_numpy(input_layer_weight.astype(np.float32)).reshape(self.model.input_layer.weight.shape[::-1]).divide(127.0).transpose(0, 1)
        self.model.input_layer.bias.data = torch.from_numpy(input_layer_bias.astype(np.float32)).reshape(self.model.input_layer.bias.shape).divide(127.0)
        
        def A1(layer, w, b, a1, a2, a3):
            a4 = a3 / a2
            #non_padded_shape = layer.weight.shape
            #padded_shape = (non_padded_shape[0], ((non_padded_shape[1]+31)//32)*32)
            weight = torch.from_numpy(w.astype(np.float32)).reshape(layer.weight.shape).divide(a4)
            bias = torch.from_numpy(b.astype(np.float32)).reshape(layer.bias.shape).divide(a3)
            #return weight[:non_padded_shape[0], :non_padded_shape[1]], bias
            return weight, bias

        kWeightScaleBits = 6
        kActivationScale = 127.0
        kBiasScale_2 = 9600.0
        kBiasScale_1 = (1 << kWeightScaleBits) * kActivationScale
        
        w1, b1 = A1(self.model.l1, weight_L1, bias_L1, kWeightScaleBits, kActivationScale, kBiasScale_1)
        w2, b2 = A1(self.model.l2, weight_L2, bias_L2, kWeightScaleBits, kActivationScale, kBiasScale_1)
        w3, b3 = A1(self.model.output_layer, weight_L3, bias_L3, kWeightScaleBits, kActivationScale, kBiasScale_2)

        self.model.l1.weight.data = w1
        self.model.l1.bias.data = b1
        self.model.l2.weight.data = w2
        self.model.l2.bias.data = b2
        self.model.output_layer.weight.data = w3
        self.model.output_layer.bias.data = b3
        return
