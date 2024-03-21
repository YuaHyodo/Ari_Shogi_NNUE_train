"""
この部分は、
https://github.com/bleu48/shogi-eval/blob/master/evaltest_nnue.py
https://github.com/bleu48/shogi-eval/blob/master/sample4-5a.py
あたりを参考に作った。
"""

import cshogi
import numpy as np 

BonaPiece_c=[1, 90, 252, 414, 576, 900, 1224, 738, 0, 738, 738, 738, 738, 1062, 1386, 0, 0, 171, 333, 495, 657, 981, 1305, 819, 0, 819, 819, 819, 819, 1143, 1467, 0, 0]
BonaPiece_h=[[1, 39, 49, 59, 69, 79, 85], [20, 44, 54, 64, 74, 82, 88]] # NNUEのPベクトル計算用のテーブル

features_num = 125388

def make_input_features(board, x1, x2, fill_zero=False):
    if fill_zero:
        x1.fill(0)
        x2.fill(0)
    pindex, qindex = [], []
    bp = board.pieces
    k0 = None
    k1 = None
    for square, piece in enumerate(bp):
        if piece == cshogi.BKING:
            k0 = square
            if k1 is not None:
                break
        elif piece == cshogi.WKING:
            k1 = square
            if k0 is not None:
                break
    bp[k0] = 0; bp[k1] = 0
    for square, piece in enumerate(bp):
        if piece > 0:
            pindex.append(BonaPiece_c[piece] + square)
            qindex.append(BonaPiece_c[piece ^ 16] + 80-square)

    pieces_in_hand = board.pieces_in_hand
    for color in cshogi.COLORS:
        for piece in cshogi.HAND_PIECES:
            piece_count = pieces_in_hand[color][piece]
            if piece_count:
                index = BonaPiece_h[color][piece]
                index2 = BonaPiece_h[color ^ 1][piece]
                for i in range(piece_count):
                    pindex.append(index + i)
                    qindex.append(index2 + i)

    if board.turn == cshogi.BLACK:
        for i in range(38):
            x1[k0*1548+pindex[i]] = 1
            x2[(80-k1)*1548+qindex[i]] = 1
    else:
        for i in range(38):
            x1[(80-k1)*1548+qindex[i]] = 1
            x2[k0*1548+pindex[i]] = 1
