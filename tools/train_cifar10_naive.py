import matplotlib
matplotlib.use('Agg')

import argparse, time, logging

import numpy as np
import mxnet as mx
import os
import datetime

from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

import gluoncv
from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, TrainingHistory
from gluoncv.data import transforms as gcv_transforms

import _init_paths
from model.resnet import resnet18_v1
from data.datasets import CIFAR10Split, CORE50Split, merge_datasets
from model.cifar_resnet20_v1 import cifar_resnet20_v1
from data.sampler import *
from mxnet.gluon.data.vision.datasets import CIFAR10

def get_opt():
    parser = argparse.ArgumentParser(description='Train a model for image classification on CIFAR10.')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='training batch size per device (CPU/GPU).')
    parser.add_argument('--inc-batch-size', type=int, default=100,
                        help='training batch size for continuous sessions.')
    parser.add_argument('--gpus', type=str, default='0,1',
                        help='gpu ids to use.')
    parser.add_argument('--model', type=str, default='cifar_resnet20_v1',
                        help='model to use. options are cifar_resnet20_v1 and cifar_quick_c4')
    parser.add_argument('--data-aug', action='store_true',
                        help='use data augmentation or not (default).')
    parser.add_argument('--initializer', type=str, default='xaiver',
                        help='initializer for the network, xaiver or prelu.')
    parser.add_argument('--sessions', type=int, default=5,
                        help='sessions for training data partition.')
    parser.add_argument('--data-sampler', type=int, default=0,
                        help='data sampler 0~4 for different data sampling strategies.')
    parser.add_argument('--data-k', type=int, default=1000,
                        help='sampling k examples by data sampler.')
    parser.add_argument('--cumulative', action='store_true')
    parser.add_argument('--plc-loss', action='store_true',
                        help='use pillar loss or not (default).')
    parser.add_argument('--cpl-loss', action='store_true',
                        help='use pillar loss or not (default).')
    parser.add_argument('--pillar-sampler', type=int, default=0,
                        help='pillar sampler 0 for different pillar generation strategies.')
    parser.add_argument('--pillar-k', type=int, default=225,
                        help='sampling k pillars by pillar sampler')
    parser.add_argument('--w1', type=float, default=1,
                        help='hyper-parameter \lambda_1 for pillar loss')
    parser.add_argument('--w2', type=float, default=1,
                        help='hyper-parameter \lambda_2 for inc loss')
    parser.add_argument('--restart', action='store_true',
                        help='restart for finetuning on each session.')
    parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=4, type=int,
                        help='number of preprocessing workers')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of training epochs for initial session.')
    parser.add_argument('--inc-epochs', type=int, default=10,
                        help='number of training epochs for continuous sessions.')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate for initial session.')
    parser.add_argument('--inc-lr', type=float, default=0.005,
                        help='learning rate for continuous sessions')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum value for optimizer, default is 0.9.')
    parser.add_argument('--wd', type=float, default=0.0005,
                        help='weight decay rate for initial session.')
    parser.add_argument('--inc-wd', type=float, default=0.0001,
                        help='weight decay rate for continuous sessions.')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--inc-lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate for continuous sessions.')
    parser.add_argument('--lr-decay-epoch', type=str, default='15',
                        help='epochs at which learning rate decays. default is 15.')
    parser.add_argument('--inc-lr-decay-epoch', type=str, default='8',
                        help='epochs at which learning rate decays for continuous sessions.')
    parser.add_argument('--mode', type=str,
                        help='mode in which to train the model. options are imperative, hybrid')
    parser.add_argument('--name', type=str, default='default',
                        help='name for experiment setting.')
    parser.add_argument('--save-dir', type=str, default='params',
                        help='directory of saved models')
    parser.add_argument('--resume-from', type=str,
                        help='resume training from the model')
    parser.add_argument('--save-plot-dir', type=str, default='.',
                        help='the path to save the history plot')
    parser.add_argument('--dataset', type=str, default='CIFAR10Split',
                        help='dataset to use. options are CIFAR10Split and CORE50Split')
    parser.add_argument('--classes', type=int, default=10,
                        help='classes for dataset')
    parser.add_argument('--resume-s1', action='store_true')
    parser.add_argument('--use_som', action='store_true')
    return parser.parse_args()

opt = get_opt()
classes = opt.classes

DatasetSplit = eval(opt.dataset)

sessions = opt.sessions

epochs, inc_epochs = opt.epochs, opt.inc_epochs
epochs = [epochs] + [inc_epochs] * (sessions - 1)

batch_size, inc_batch_size = opt.batch_size, opt.inc_batch_size

restart = opt.restart

gpu_ids = [int(i) for i in opt.gpus.strip().split(',')]
batch_size *= max(1, len(gpu_ids))
inc_batch_size *= max(1, len(gpu_ids))
context = [mx.gpu(i) for i in gpu_ids] if len(gpu_ids) > 0 else [mx.cpu()]
num_workers = opt.num_workers
use_cpl = opt.cpl_loss
batch_sizes = [batch_size] + [inc_batch_size] * (sessions - 1)

lr_decay, inc_lr_decay = opt.lr_decay, opt.inc_lr_decay
lr_decay_epoch = [int(i) for i in opt.lr_decay_epoch.split(',')] + [np.inf]
inc_lr_decay_epoch = [int(i) for i in opt.inc_lr_decay_epoch.split(',')] + [np.inf]
lr_decay_epochs = [lr_decay_epoch] + [inc_lr_decay_epoch] * (sessions - 1)

model_name = opt.model
kwargs = {'classes': classes}

if model_name == 'core50_resnet18':
    net = resnet18_v1(pretrained=True, ctx=context)
    net.output = gluon.nn.Dense(classes)
    net.output.initialize(mx.init.Xavier(), ctx=context)
elif model_name == 'cifar_resnet20_v1':
    net = cifar_resnet20_v1()
    feature_size = 64
else:
    raise KeyError

if opt.resume_from:
    net.load_parameters(opt.resume_from, ctx=context)

optimizer = 'sgd'
optimizer_param = {
    'learning_rate': opt.lr, 'wd': opt.wd,
    'momentum': opt.momentum, 'clip_gradient': 2
}

inc_optimizer_param = {
    'learning_rate': opt.inc_lr, 'wd': opt.inc_wd,
    'momentum': opt.momentum, 'clip_gradient': 5
}
optimizer_params = [optimizer_param] + [inc_optimizer_param] * (sessions - 1)

initializer = mx.init.Xavier() if opt.initializer == 'xavier' else mx.init.MSRAPrelu()

save_dir = opt.save_dir
if not os.path.exists(save_dir):
    makedirs(save_dir)
name = opt.name

plot_path = opt.save_plot_dir

logging.basicConfig(level=logging.INFO)
logging.info(opt)

trans, aug_trans = list(), list()
if opt.data_aug:
    aug_trans = [gcv_transforms.RandomCrop(32, pad=4),
                 transforms.RandomFlipLeftRight()]
trans.append(transforms.ToTensor())
trans.append(transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]))

transform_train = transforms.Compose(aug_trans + trans)
transform_test = transforms.Compose(trans)
data_sampler = eval('DataSampler%d' % opt.data_sampler)()
num_data_samples = opt.data_k
use_som = opt.use_som
use_pillars = opt.plc_loss
pillar_sampler = eval('PillarSampler%d' % opt.pillar_sampler)()
num_pillar_samples = opt.pillar_k
w1 = opt.w1
w2 = opt.w2

cumulative = opt.cumulative

def get_dataloader(dataset, batch_size, train=True):
    if train:
        dataloader = gluon.data.DataLoader(
            dataset.transform_first(transform_train), batch_size=batch_size,
            shuffle=True, last_batch='discard', num_workers=num_workers
        )
    else:
        dataloader = gluon.data.DataLoader(
            dataset.transform_first(transform_test), batch_size=batch_size,
            shuffle=False, num_workers=num_workers
        )
    return dataloader

def test(net, ctx, val_data):
    metric = mx.metric.Accuracy()
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        outputs = [net(X)[1] for X in data]
        metric.update(label, outputs)
    return metric.get()

def compute_average_acc(acc_list):
    mean = np.mean(acc_list)
    std = np.std(acc_list)
    return mean, std

def mx_calculate_dist(pillar, positive):
    d1 = mx.ndarray.sum(pillar * pillar, axis=1).reshape(1,1)
    d2 = mx.ndarray.sum(positive * positive, axis=1).reshape(-1,1)
    eps = 1e-12
    a = d1.repeat(int(positive.shape[0]))
    b = mx.ndarray.transpose(d2.repeat(1))
    c = 2.0 * mx.ndarray.dot(pillar, mx.ndarray.transpose(positive))
    return mx.ndarray.sqrt(mx.ndarray.abs((a + b - c))+eps)

def calculate_positive(output, pillars, label, pillar_label):
    # output: NDarray feature on 1 gpu
    # pillar: list on num gpus
    pillars = mx.nd.concatenate(pillars, 0)
    pillars = mx.nd.array(pillars, ctx=output.context)
    pillar_label = mx.nd.concatenate(pillar_label, 0)
    label = label.asnumpy()
    pillar_label = pillar_label.asnumpy()
    min_index = list()
    for i in range(output.shape[0]):
        gt_index = np.where(label[i] == pillar_label)
        if gt_index[0].shape[0]!=0:
            dist = mx_calculate_dist(output[i].reshape(1,-1),pillars[gt_index].reshape(-1,64))
            min_index.append(gt_index[0][int(mx.nd.argmin(dist,1).asnumpy()[0])])
        else:
            min_index.append(-1)
    return min_index

def train(net, ctx):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    net.initialize(initializer, ctx=ctx)

    val_dataloader = get_dataloader(DatasetSplit(train=False), batch_size=100, train=False)

    metric = mx.metric.Accuracy()
    train_metric = mx.metric.Accuracy()
    loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
    if use_pillars:
        plc_loss_fn = gluon.loss.L2Loss(weight=w1)
    if use_cpl:
        loss_fn_cpl = gluon.loss.L2Loss(weight=w2)
    train_history = TrainingHistory(['training-error', 'validation-error'])
    timestr = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    param_dir = os.path.join(save_dir, name, timestr)
    if not os.path.exists(param_dir):
        os.makedirs(param_dir)

    param_file_fmt = '%s/cifar10_%s_%d-%d-%d.params'
    training_record_fmt = '[Session %d, Epoch %d] train=%.4f val=%.4f loss=%.4f '
    if use_pillars:
        training_record_fmt += 'plc-loss=%.4f '
    training_record_fmt += 'time: %.2f'

    prev_dataloader, prev_dataset, prev_pillarset, pillarset = None, None, None, None
    record_acc = dict()

    for sess in range(sessions):

        record_acc[sess] = list()
        logging.info("[Session %d] begin training ..." % (sess+1))
        if sess == 0 and opt.resume_s1:
            _, val_acc = test(net, ctx, val_dataloader)
            record_acc[sess].append(val_acc)
            logging.info('session 1 test acc : %.4f'% val_acc)
            prev_dataset = DatasetSplit(split_id=sess, train=True)
            prev_dataloader = get_dataloader(prev_dataset, batch_sizes[sess], train=True)
            continue

        train_dataset = DatasetSplit(split_id=sess, train=True)
        lr_decay_count, best_val_score = 0, 0

        if sess != 0:
            # Sampling data for continuous training
            logging.info("[Session %d] sampling training data and pillars ..." % (sess+1))
            dataloader = get_dataloader(train_dataset, batch_size=100, train=False)
            train_dataset = data_sampler.sample_dataset(
                train_dataset, dataloader, net, loss_fn, num_data_samples, ctx=ctx)
            if cumulative:
                train_dataset = merge_datasets(prev_dataset, train_dataset)

        train_dataloader = get_dataloader(train_dataset, batch_sizes[sess], train=True)
        # Build trainer for net.
        trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params[sess])

        for epoch in range(epochs[sess]):
            tic = time.time()
            train_metric.reset()
            metric.reset()
            train_loss, train_plc_loss = 0, 0
            num_batch = len(train_dataloader)

            if epoch == lr_decay_epochs[sess][lr_decay_count]:
                trainer.set_learning_rate(trainer.learning_rate*lr_decay)
                lr_decay_count += 1

            for i, batch in enumerate(train_dataloader):
                data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
                label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
                all_loss = list()
                with ag.record():
                    output = [net(X)[1] for X in data]
                    output_feat = [net(X)[0] for X in data]
                    loss = [loss_fn(yhat, y) for yhat, y in zip(output, label)]
                    all_loss.extend(loss)
                    # Normalize each loss for the trainer with batch_size=1
                    all_loss = [nd.mean(l) for l in all_loss]

                ag.backward(all_loss)
                trainer.step(1, ignore_stale_grad=True)
                train_loss += sum([l.sum().asscalar() for l in loss])
                if sess > 0 and use_pillars:
                    train_plc_loss += sum([al.mean().asscalar() for al in plc_loss])

                train_metric.update(label, output)

            train_loss /= batch_sizes[sess] * num_batch
            _, acc = train_metric.get()
            _, val_acc = test(net, ctx, val_dataloader)
            train_history.update([1-acc, 1-val_acc])
            train_history.plot(save_path='%s/%s_history.png'%(plot_path, model_name))
            if epoch >= epochs[sess] - 5:
                record_acc[sess].append(val_acc)

            training_record = [sess+1, epoch, acc, val_acc, train_loss]
            if use_pillars:
                training_record += [train_plc_loss]
            training_record += [time.time() - tic]
            logging.info(training_record_fmt % tuple(training_record))

            net.save_parameters(param_file_fmt % (param_dir, model_name, sess, epochs[sess], epoch))
        prev_dataset = train_dataset
        prev_dataloader = train_dataloader
        prev_pillarset = pillarset
        if sess == 0 or sess == 1:
            save_data = get_dataloader(DatasetSplit(split_id=0, train=True), batch_size=10000, train=True)
            for i, batch in enumerate(save_data):
                data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
                label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
                outputs = net(data[0])[0]
                np.save('session{}_feats.npy'.format(sess),outputs.asnumpy())
                np.save('session{}_label.npy'.format(sess),label[0].asnumpy())

    for i in range(len(list(record_acc.keys()))):
        mean = np.mean(np.array(record_acc[i]))
        std = np.std(np.array(record_acc[i]))
        print('[Sess %d] Mean=%f Std=%f' % (i+1, mean, std))

    # compute_average_acc(record_acc)

def main():
    if opt.mode == 'hybrid':
        net.hybridize()

    train(net, context)

if __name__ == '__main__':
    main()
