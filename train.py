import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import Model
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import ScanNet, ModelNet, ShapeNet, label_to_idx, ScanNet_Test, ModelNet_Test, ShapeNet_Test
from torch.autograd import Variable
import numpy as np
import os
import argparse
import mmd
import math
import warnings
import log
import time

NWORKERS = 4
# ==================
# init
# ==================

warnings.filterwarnings("ignore")

# Command setting
parser = argparse.ArgumentParser(description='Main')
parser.add_argument('-source', '-s', type=str, help='source dataset', default='scannet')
parser.add_argument('-target', '-t', type=str, help='target dataset', default='modelnet')
parser.add_argument('-batchsize', '-b', type=int, help='batch size', default=16)
parser.add_argument('-gpu', '-g', type=str, help='cuda id', default='0')
parser.add_argument('-epochs', '-e', type=int, help='training epoch', default=200)
parser.add_argument('-models', '-m', type=str, help='alignment model', default='MDA')
parser.add_argument('-lr',type=float, help='learning rate', default=0.0001)
parser.add_argument('-scaler',type=float, help='scaler of learning rate', default=1.)
parser.add_argument('-weight',type=float, help='weight of src loss', default=1.)
parser.add_argument('-datadir',type=str, help='directory of data', default='./')
parser.add_argument('-tb_log_dir', type=str, help='directory of tb', default='./')
args = parser.parse_args()

io = log.IOStream(args)
io.cprint(str(args))

if not os.path.exists(os.path.join(os.getcwd(), args.tb_log_dir)):
    os.makedirs(os.path.join(os.getcwd(), args.tb_log_dir))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = args.batchsize * len(args.gpu.split(','))
LR = args.lr
weight_decay = 5e-4
momentum = 0.9
max_epoch = args.epochs
num_class = 10
dir_root = os.path.join(args.datadir, 'PointDA_data/')

# ==================
# Read Data
# ==================
def split_set(dataset, domain, set_type="source"):
    """
    Input:
        dataset
        domain - modelnet/shapenet/scannet
        type_set - source/target
    output:
        train_sampler, valid_sampler
    """
    train_indices = dataset.train_ind
    val_indices = dataset.val_ind
    unique, counts = np.unique(dataset.label[train_indices], return_counts=True)
    print("Occurrences count of classes in " + set_type + " " + domain +
              " train part: " + str(dict(zip(unique, counts))))
    unique, counts = np.unique(dataset.label[val_indices], return_counts=True)
    print("Occurrences count of classes in " + set_type + " " + domain +
              " validation part: " + str(dict(zip(unique, counts))))
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    return train_sampler, valid_sampler

src_dataset = args.source
trgt_dataset = args.target
data_func = {'modelnet': ModelNet, 'scannet': ScanNet, 'shapenet': ShapeNet}
data_test_func = {'modelnet': ModelNet_Test, 'scannet': ScanNet_Test, 'shapenet': ShapeNet_Test}

src_trainset = data_func[src_dataset](io, args.datadir, 'train')
trgt_trainset = data_func[trgt_dataset](io, args.datadir, 'train')
trgt_testset = data_test_func[trgt_dataset](io, args.datadir, 'test')

# Creating data indices for training and validation splits:
src_train_sampler, src_valid_sampler = split_set(src_trainset, src_dataset, "source")
trgt_train_sampler, trgt_valid_sampler = split_set(trgt_trainset, trgt_dataset, "target")

# dataloaders for source and target
src_train_loader = DataLoader(src_trainset, num_workers=NWORKERS, batch_size=args.batchsize,
                               sampler=src_train_sampler, drop_last=True)
src_val_loader = DataLoader(src_trainset, num_workers=NWORKERS, batch_size=args.batchsize,
                             sampler=src_valid_sampler)
trgt_train_loader = DataLoader(trgt_trainset, num_workers=NWORKERS, batch_size=args.batchsize,
                                sampler=trgt_train_sampler, drop_last=True)
trgt_val_loader = DataLoader(trgt_trainset, num_workers=NWORKERS, batch_size=args.batchsize,
                                  sampler=trgt_valid_sampler)
trgt_test_loader = DataLoader(trgt_testset, num_workers=NWORKERS, batch_size=args.batchsize)

# ==================



model = Model.Net_MDA()
model = model.to(device=device)

criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device=device)

remain_epoch=50

# Optimizer

params = [{'params':v} for k,v in model.g.named_parameters() if 'pred_offset' not in k]

optimizer_g = optim.Adam(params, lr=LR, weight_decay=weight_decay)
lr_schedule_g = optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=args.epochs+remain_epoch)

optimizer_c = optim.Adam([{'params':model.c1.parameters()},{'params':model.c2.parameters()}], lr=LR*2,
                            weight_decay=weight_decay)
lr_schedule_c = optim.lr_scheduler.CosineAnnealingLR(optimizer_c, T_max=args.epochs+remain_epoch)

optimizer_dis = optim.Adam([{'params':model.g.parameters()},{'params':model.attention_s.parameters()},{'params':model.attention_t.parameters()}], 
    lr=LR*args.scaler, weight_decay=weight_decay)
lr_schedule_dis = optim.lr_scheduler.CosineAnnealingLR(optimizer_dis, T_max=args.epochs+remain_epoch)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by half by every 5 or 10 epochs"""
    if epoch > 0:
        if epoch <= 30:
            lr = args.lr * args.scaler * (0.5 ** (epoch // 5))
        else:
            lr = args.lr * args.scaler * (0.5 ** (epoch // 10))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def discrepancy(out1, out2):
    """discrepancy loss"""
    out = torch.mean(torch.abs(F.softmax(out1, dim=-1) - F.softmax(out2, dim=-1)))
    return out

def make_variable(tensor, volatile=False):
    """Convert Tensor to Variable."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor, volatile=volatile)

best_target_test_acc = 0

for epoch in range(max_epoch):
    time_start = time.time()
        
    lr_schedule_g.step(epoch=epoch)
    lr_schedule_c.step(epoch=epoch)
    adjust_learning_rate(optimizer_dis, epoch)

    model.train()

    loss_total = 0
    loss_adv_total = 0
    loss_node_total = 0
    correct_total = 0
    data_total = 0
    data_t_total = 0
    cons = math.sin((epoch + 1)/max_epoch * math.pi/2 )

    # Training

    for batch_idx, (batch_s, batch_t) in enumerate(zip(src_train_loader, trgt_train_loader)):

        data, label = batch_s[0], batch_s[1]
        data_t, label_t = batch_t[0], batch_t[1]

        data = data.to(device=device).permute(0, 2, 1).unsqueeze(-1)
        label = label.to(device=device).long()
        data_t = data_t.to(device=device).permute(0, 2, 1).unsqueeze(-1)
        label_t = label_t.to(device=device).long()

        pred_s1,pred_s2 = model(data)
        pred_t1,pred_t2 = model(data_t, constant = cons, adaptation=True)

        # Classification loss

        loss_s1 = criterion(pred_s1, label)
        loss_s2 = criterion(pred_s2, label)

        # Adversarial loss

        loss_adv = - 1 * discrepancy(pred_t1, pred_t2) 

        loss_s = loss_s1  +  loss_s2
        loss = args.weight * loss_s + loss_adv

        loss.backward()
        optimizer_g.step()
        optimizer_c.step()
        optimizer_g.zero_grad()
        optimizer_c.zero_grad()


        # Local Alignment
        
        feat_node_s = model(data, node_adaptation_s=True)
        feat_node_t = model(data_t, node_adaptation_t=True)
        sigma_list = [0.01, 0.1, 1, 10, 100]
        loss_node_adv = 1 * mmd.mix_rbf_mmd2(feat_node_s, feat_node_t, sigma_list)
        loss = loss_node_adv

        loss.backward()
        optimizer_dis.step()
        optimizer_dis.zero_grad()

        loss_total += loss_s.item() * data.size(0)
        loss_adv_total += loss_adv.item() * data.size(0)
        loss_node_total +=  loss_node_adv.item() * data.size(0)
        data_total += data.size(0)
        data_t_total += data_t.size(0)

    
    time_end = time.time()
    print("Epoch time: %.5f hours" % ((time_end - time_start) / 3600))


    # Testing

    with torch.no_grad():
        model.eval()
        loss_total = 0
        correct_total = 0
        data_total = 0
        acc_class = torch.zeros(10,1)
        acc_to_class = torch.zeros(10,1)
        acc_to_all_class = torch.zeros(10,10)

        for batch_idx, data in enumerate(trgt_val_loader):
            data = data[0].to(device=device)
            label = data[1].to(device=device).long()
            pred1, pred2 = model(data)
            output = (pred1 + pred2)/2
            loss = criterion(output, label)
            _, pred = torch.max(output, 1)

            loss_total += loss.item() * data.size(0)
            correct_total += torch.sum(pred == label)
            data_total += data.size(0)

        pred_loss = loss_total/data_total
        pred_acc = correct_total.double()/data_total

        if pred_acc > best_target_test_acc:
            best_target_test_acc = pred_acc
        print ('Target 1:{} [overall_acc: {:.4f} \t loss: {:.4f} \t Best Target Acc: {:.4f}]'.format(
        epoch, pred_acc, pred_loss, best_target_test_acc
        ))



if __name__ == '__main__':
    since = time.time()
    main()
    time_pass = since - time.time()
    print('Training complete in {:.0f}h'.format(time_pass / 3600))

