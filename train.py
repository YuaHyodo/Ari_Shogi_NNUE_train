"""
python-dlshogi2のtrain.py(https://github.com/TadaoYamaoka/python-dlshogi2/blob/main/pydlshogi2/train.py)をベース(にしているAri Shogiの学習部をベース)にしている
elmo式に関する部分は、dlshogiのtrain.py( https://github.com/TadaoYamaoka/DeepLearningShogi/blob/master/dlshogi/train.py )を参考にしている。
"""
from NNUE_network1 import*
from dataloader import*
import cshogi

import ranger
import torch.optim as optim
import torch

from datetime import datetime as dt
import argparse
import codecs
import gc

parser = argparse.ArgumentParser()
parser.add_argument('train_data', type=str, nargs='+')
parser.add_argument('test_data', type=str)
parser.add_argument('--resume', '-r', type=str, default='')
parser.add_argument('--gpu', '-g', type=int, default=0)
parser.add_argument('--epoch', '-e', type=int, default=1)
parser.add_argument('--batchsize', '-b', type=int, default=1024)
parser.add_argument('--test_batchsize', '-t', type=int, default=1024)

#elmo式における勝敗項と勝率項のバランスをとる部分
#評価値を使わない設定(下のeval_aが0)の場合は無視される
parser.add_argument('--val_lambda', type=float, default=0.33)
#評価値を勝率に変換するときに使う値。0に設定した場合は評価値を学習に使わない
#データセットごとに最適な値が異なるので、事前に求めておくと良い
#自分は、以下のコードを利用して求めた値をメモしておいて、それを使っている
#データローダー側で自動で求めても良いが、実行に時間かかるし、自分は上記の方法で十分なので実装してない。
#https://gist.github.com/TadaoYamaoka/07c5ca2f067741b2f01613dfcada4895
#https://tadaoyamaoka.hatenablog.com/entry/2021/05/06/213506
parser.add_argument('--eval_a', type=int, default=0)
parser.add_argument('--eval_a_test', type=int, default=0)

#AMPを使用するか(ONにする事を推奨)
parser.add_argument('--use_amp', action='store_true')
#重複局面の除去(教師データのみ)。時間かかるので事前にやっておくことを推奨。
parser.add_argument('--unique', action='store_true')

#パラメータをリセットする層を決める。ONにした層のパラメータは初期化される
parser.add_argument('--make_new_layer_input', action='store_true')
parser.add_argument('--make_new_layer_L1', action='store_true')
parser.add_argument('--make_new_layer_L2', action='store_true')
parser.add_argument('--make_new_layer_output', action='store_true')

#層毎に学習率を設定する。0に設定すると、その層は学習されない
#https://github.com/nodchip/nnue-pytorch/blob/master/model.py#L136 )では、
#出力層のパラメータの学習率だけ小さく設定されているので、そうした方が良い可能性もあるが、実際どうなのかは不明。
parser.add_argument('--lr_input', type=float, default=0.00005)
parser.add_argument('--lr_L1', type=float, default=0.00005)
parser.add_argument('--lr_L2', type=float, default=0.00005)
parser.add_argument('--lr_output', type=float, default=0.00005)

#optimizerの種類と各種設定を弄る場所
parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer(sgd or adam or adamw or ranger)')
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--sgd_momentum', type=float, default=0.9)
parser.add_argument('--sgd_nesterov', action='store_true')
parser.add_argument('--adam_eps', type=float, default=1e-8)

#学習率スケジューラーに関する設定
#CyclicLRの場合、設定された学習率をbase、その--CyclicLR_maxLR倍の値をmaxに設定する
#ReduceLROnPlateauの場合、設定された学習率を初期値に設定する
parser.add_argument('--lr_scheduler_type', type=str, default='none')#none(なし) / CyclicLR / ReduceLROnPlateau
parser.add_argument('--CyclicLR_maxLR', type=float, default=10.0)
parser.add_argument('--CyclicLR_step_size', type=int, default=2000)
parser.add_argument('--ReduceLROnPlateau_patience', type=int, default=2)
parser.add_argument('--ReduceLROnPlateau_factor', type=float, default=0.1)
parser.add_argument('--ReduceLROnPlateau_threshold', type=float, default=0.0001)

#チェックポイントのディレクトリ
parser.add_argument('--checkpoint', default='./checkpoints/')
#テストデータ1batchを使ってテストする間隔
parser.add_argument('--eval_interval', type=int, default=5000)
#ログファイル
parser.add_argument('--log', type=str, default='NNUE_train_test.txt')

#何epochに1回セーブするか( 何epochも回すとき、1epoch毎にセーブされるファイルでストレージが圧迫されるので、その時に使う)
parser.add_argument('--save_model_interval', type=int, default=1)

#1epoch中にバックアップを取る場合のオプション
#( GoogleColabみたいな環境で使うときや1epochが長い場合に使うと悲しい事故を防げるかもしれない )
parser.add_argument('--backup_interval', type=int, default=5000)
parser.add_argument('--backup_dire', default='./backup/')

#学習前にテストデータすべてを使ってテストするオプション
parser.add_argument('--test_before_train', action='store_true')
args = parser.parse_args()

os.makedirs(args.checkpoint + 'output/', exist_ok=True)
os.makedirs(args.backup_dire + 'output/', exist_ok=True)

def print_kai(w):
    w = str(dt.now()) + ' | ' + w
    print(w)
    if '.txt' not in args.log:
        return
    print(w, file=codecs.open(args.log, 'a', 'utf-8'))
    return

print_kai('args = {}'.format(args))

if args.gpu >= 0:
    device = torch.device('cuda:{}'.format(args.gpu))
else:
    device = torch.device('cpu')

bce_with_logits_loss = torch.nn.BCEWithLogitsLoss()
scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

load_ckpt = False
if args.resume == '':
    print_kai('make_new_model'.format(args.resume))
    model = NNUE_halfkp(L1=256, L2=32, L3=32)
else:
    print_kai('Loading the checkpoint from {}'.format(args.resume))
    if '.bin' in args.resume:
        print_kai('Load from .bin')
        with open(args.resume, 'rb') as f:
            reader = NNUEReader(f)
        model = reader.model
    else:
        print_kai('Load from checkpoint')
        load_ckpt = True
        ckpt = torch.load(args.resume, map_location=device)
        model = NNUE_halfkp(L1=256, L2=32, L3=32)
        
#設定に基づいてパラメータを作り直す
if args.make_new_layer_input:
    print_kai('input_layer is included in the layer to be initialized.')
    model.input_layer = nn.Linear(features_num, model.units[0])
if args.make_new_layer_L1:
    print_kai('L1_layer is included in the layer to be initialized.')
    model.l1 = nn.Linear(model.units[0] * 2, model.units[1])
if args.make_new_layer_L2:
    print_kai('L2_layer is included in the layer to be initialized.')
    model.l2 = nn.Linear(model.units[1], model.units[2])
if args.make_new_layer_output:
    print_kai('output_layer is included in the layer to be initialized.')
    model.output_layer = nn.Linear(model.units[2], 1)

model.to(device)
if load_ckpt:
    print_kai('load_state_dict')
    model.load_state_dict(ckpt['model'])

#学習対象の層のパラメータのみをoptimizerに渡す
params_list = []
if args.lr_input != 0:
    print_kai('input_layer is set to be trained. LR is {}'.format(args.lr_input))
    params_list.append({'params': [model.input_layer.weight, model.input_layer.bias], 'lr': args.lr_input})

if args.lr_L1 != 0:
    print_kai('L1_layer is set to be trained. LR is {}'.format(args.lr_L1))
    params_list.append({'params': [model.l1.weight, model.l1.bias], 'lr': args.lr_L1})

if args.lr_L2 != 0:
    print_kai('L2_layer is set to be trained. LR is {}'.format(args.lr_L2))
    params_list.append({'params': [model.l2.weight, model.l2.bias], 'lr': args.lr_L2})
    
if args.lr_output != 0:
    print_kai('output_layer is set to be trained. LR is {}'.format(args.lr_output))
    params_list.append({'params': [model.output_layer.weight, model.output_layer.bias], 'lr': args.lr_output})

if load_ckpt:
    #checkpointからの読み込みの場合
    args.optimizer = ckpt['opt_type']
    args.lr_scheduler_type = ckpt['lr_scheduler_type']
    
    print_kai('optimizer_type = {}'.format(ckpt['opt_type']))
    if ckpt['opt_type'] == 'sgd':
        optimizer = optim.SGD(params_list, momentum=args.sgd_momentum, weight_decay=args.weight_decay, nesterov=args.sgd_nesterov)
    elif ckpt['opt_type'] == 'adamw':
        optimizer = optim.AdamW(params_list, weight_decay=args.weight_decay, eps=args.adam_eps)
    elif ckpt['opt_type'] == 'ranger':
        optimizer = ranger.Ranger(params_list, weight_decay=args.weight_decay, eps=args.adam_eps)
    else:
        optimizer = optim.Adam(params_list, weight_decay=args.weight_decay, eps=args.adam_eps)
    optimizer.load_state_dict(ckpt['optimizer'])
    for index, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = params_list[index]['lr']
        param_group['weight_decay'] = args.weight_decay
    
    if args.use_amp and 'scaler' in ckpt:
        scaler.load_state_dict(ckpt['scaler'])
    
    if ckpt['lr_scheduler_type'] == 'CyclicLR':
        print_kai('LR_scheduler = CyclicLR')
        base_lr_list = [i['lr'] for i in params_list]
        max_lr_list = [i * args.CyclicLR_maxLR for i in base_lr_list]
        print_kai('base_lr: {} / max_lr: {}'.format(base_lr_list, max_lr_list))
        lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr_list, 
                        max_lr=max_lr_list, step_size_up=args.CyclicLR_step_size, step_size_down=None)
    elif ckpt['lr_scheduler_type'] == 'ReduceLROnPlateau':
        print_kai('LR_scheduler = ReduceLROnPlateau')
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.ReduceLROnPlateau_factor, 
            patience=args.ReduceLROnPlateau_patience, threshold=args.ReduceLROnPlateau_threshold)
        lr_scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])
    else:
        print_kai('LR_scheduler = None')
else:
    #.binからの読み込み or 新規生成の場合
    print_kai('optimizer_type = {}'.format(args.optimizer))
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(params_list, momentum=args.sgd_momentum, weight_decay=args.weight_decay, nesterov=args.sgd_nesterov)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(params_list, weight_decay=args.weight_decay, eps=args.adam_eps)
    elif args.optimizer == 'ranger':
        optimizer = ranger.Ranger(params_list, weight_decay=args.weight_decay, eps=args.adam_eps)
    else:
        optimizer = optim.Adam(params_list, weight_decay=args.weight_decay, eps=args.adam_eps)
    if args.lr_scheduler_type == 'CyclicLR':
        print_kai('LR_scheduler = CyclicLR')
        base_lr_list = [i['lr'] for i in params_list]
        max_lr_list = [i * args.CyclicLR_maxLR for i in base_lr_list]
        print_kai('base_lr: {} / max_lr: {}'.format(base_lr_list, max_lr_list))
        lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr_list, 
                        max_lr=max_lr_list, step_size_up=args.CyclicLR_step_size, step_size_down=None)
    elif args.lr_scheduler_type == 'ReduceLROnPlateau':
        print_kai('LR_scheduler = ReduceLROnPlateau')
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.ReduceLROnPlateau_factor, 
            patience=args.ReduceLROnPlateau_patience, threshold=args.ReduceLROnPlateau_threshold)
    else:
        print_kai('LR_scheduler = None')

train_data_files = dire_list_to_files(args.train_data)
N = 10
if len(train_data_files) < N:
    train_data = [train_data_files]
else:
    train_data = []
    for i in range((len(train_data_files) // N)+1):
        F = train_data_files[i * N:(i+1)*N]
        if len(F) >= 2:
            train_data.append(F)
        elif len(F) == 1:
            if len(train_data) > 0:
                train_data[-1].extend(F)
            else:
                train_data.append(F)
print_kai('train_data_files: {}'.format(train_data))
test_data_files = dire_list_to_files(args.test_data)
print_kai('test_data_files: {}'.format(test_data_files))
test_dataloader = HcpeDataLoader(test_data_files, args.test_batchsize, device,
                                 eval_a=args.eval_a_test, unique=False)
print_kai('len(test_dataloader) = {}'.format(len(test_dataloader)))

def binary_accuracy(y, t):
    pred = y >= 0
    truth = t >= 0.5
    return pred.eq(truth).sum().item() / len(t)

def save_model(m, opt, sca, e, dire=None):
    ckpt = {'model': m.state_dict(),
            'optimizer': opt.state_dict(),
            'scaler': sca.state_dict(),
            'opt_type': args.optimizer,
            'lr_scheduler_type': args.lr_scheduler_type}
    if args.lr_scheduler_type not in ('ReduceLROnPlateau'):
        ckpt['lr_scheduler_state_dict'] = 0
    else:
        ckpt['lr_scheduler_state_dict'] = lr_scheduler.state_dict()
    if dire is None:
        dire = args.checkpoint
    torch.save(ckpt, dire + 'ckpt_epoch_{}.pth'.format(e))

    m.to(torch.device('cpu'))
    w = NNUEWriter(m)
    w.write_to_file(dire + 'output/epoch_{}.bin'.format(e))
    m.to(device)
    return

def eval_model(model, dataloader, batch_num=None):
    model.eval()
    if batch_num == 1:
        with torch.no_grad():
            x1, x2, result, value = dataloader.sample()
            y = model(x1, x2)
            if dataloader.eval_a == 0:
                test_loss = bce_with_logits_loss(y, result).item()
            else:
                loss1 = bce_with_logits_loss(y, result)
                loss2 = bce_with_logits_loss(y, value)
                loss = (loss1 * (1.0 - args.val_lambda) + loss2 * args.val_lambda)
                test_loss = loss.item()
            test_acc = binary_accuracy(y, result)
        return test_loss, test_acc
    test_steps = 0
    test_loss = 0
    test_acc = 0
    for i, (x1, x2, result, value) in enumerate(dataloader):
        test_steps += 1
        with torch.no_grad():
            y = model(x1, x2)
            if dataloader.eval_a == 0:
                #評価値を使わない場合
                test_loss = bce_with_logits_loss(y, result).item()
            else:
                loss1 = bce_with_logits_loss(y, result)
                loss2 = bce_with_logits_loss(y, value)
                loss = (loss1 * (1.0 - args.val_lambda) + loss2 * args.val_lambda)
                test_loss += loss.item()
            test_acc += binary_accuracy(y, result)
        if batch_num is not None and i >= batch_num:
            break
        return test_loss / test_steps, test_acc / test_steps

if args.test_before_train:
    print_kai('test_before_train == True')
    test_loss, test_acc = eval_model(model, test_dataloader)
    print_kai('test_before_train | test_loss = {}, test_acc = {}'.format(test_loss, test_acc))
    
for e in range(args.epoch):
    print_kai('start epoch {}'.format(e))
    steps_interval = 0
    sum_loss_interval = 0
    steps_epoch = 0
    sum_loss_epoch = 0
    steps_interval_b = 0
    for train_data_C in train_data:
        train_dataloader = HcpeDataLoader(train_data_C, args.batchsize, device,
                    shuffle=True, eval_a=args.eval_a, unique=args.unique)
        for x1, x2, result, value in train_dataloader:
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                model.train()
                y = model(x1, x2)
                if train_dataloader.eval_a == 0:
                    #評価値を使わない場合
                    loss = bce_with_logits_loss(y, result)
                else:
                    loss1 = bce_with_logits_loss(y, result)
                    loss2 = bce_with_logits_loss(y, value)
                    loss = (loss1 * (1.0 - args.val_lambda) + loss2 * args.val_lambda)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if args.lr_scheduler_type == 'CyclicLR':
                    lr_scheduler.step()
            #
            steps_epoch += 1
            steps_interval += 1
            steps_interval_b += 1
            sum_loss_epoch += loss.item()
            sum_loss_interval += loss.item()
            gc.collect()
            if steps_interval_b % args.backup_interval == 0:
                print_kai('save backup')
                save_model(model, optimizer, scaler, str(e) + '_steps_' + str(steps_interval_b), dire=args.backup_dire)
            if steps_epoch % args.eval_interval == 0:
                test_loss, test_acc = eval_model(model, test_dataloader, batch_num=1)
                print_kai('epoch = {}, steps = {}, train_loss = {}, test_loss = {}, test_acc: {}'.format(e, steps_epoch,
                    sum_loss_interval / steps_interval, test_loss, test_acc))
                sum_loss_interval = 0
                steps_interval = 0
                gc.collect()
    #
    if e % args.save_model_interval == 0:
        print_kai('save checkpoint')
        save_model(model, optimizer, scaler, e)
    #
    print_kai('start eval model')
    test_loss, test_acc = eval_model(model, test_dataloader)
    print_kai('epoch = {}, steps = {}, train_loss_avr = {}, test_loss = {}, test_acc = {}'.format(e, steps_epoch,
        sum_loss_epoch / steps_epoch, test_loss, test_acc))
    if args.lr_scheduler_type == 'ReduceLROnPlateau':
        lr_scheduler.step(test_loss)
    print_kai('finish_epoch {}'.format(e))

print_kai('save model')
save_model(model, optimizer, scaler, 'last')

#.binファイルに出力したモデルを読み込んで再テストする
print_kai('start eval model after quantization')
with open(args.checkpoint + 'output/epoch_{}.bin'.format('last'), 'rb') as f:
    reader = NNUEReader(f)
model = reader.model
model.to(device)

test_loss, test_acc = eval_model(model, test_dataloader)
print_kai('test after quantization | test_loss = {}, test_acc = {}'.format(test_loss, test_acc))
