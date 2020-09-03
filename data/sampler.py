import mxnet as mx
from mxnet import gluon
import numpy as np
import numpy.random as npr
from tqdm import tqdm

from .datasets import SampleDataset, Imagenet_SampleDataset, merge_datasets
import random

def get_full_data(dataloader):
    data_list, label_list = list(), list()
    for i, batch in enumerate(dataloader):
        d, l = batch
        data_list.append(d)
        label_list.append(l)
    data = mx.nd.concat(*data_list, dim=0)
    label = mx.nd.concat(*label_list, dim=0)
    return data, label

# Data sampler
class DataSampler():
    def __init__(self):
        self._mini = 100
        pass

    def sample_dataset(self, **kwargs):
        raise Exception("To be implemented in the sub-class.")


# R0 - use all data
class DataSampler0(DataSampler):
    def __init__(self):
        super(DataSampler0, self).__init__()

    def sample_dataset(self, train_dataset, dataloader, net, loss_fn, k, ctx):
        return train_dataset

# R1 - hard example sampler
class DataSampler1(DataSampler):
    def __init__(self):
        super(DataSampler1, self).__init__()

    def sample_dataset(self, dataset, dataloader, net, loss_fn, k, ctx):
        losses = list()
        for i, batch in tqdm(enumerate(dataloader)):
            d = batch[0].as_in_context(ctx[0])
            l = batch[1].as_in_context(ctx[0])
            loss = loss_fn(net(d)[1], l)
            losses.append(loss)
        losses = mx.nd.concat(*losses, dim=0)
        losses = losses.asnumpy()
        sample_inds = np.argsort(losses)[-k:]
        if dataset.name == 'ImageNetSplit':
            return Imagenet_SampleDataset(dataset, sample_inds)
        elif dataset.name == 'CIFAR10Split' or 'CIFAR100Split' or 'CORE50Split':
            return SampleDataset(dataset, sample_inds)


# R2 - error sampler
class DataSampler2(DataSampler):
    def __init__(self):
        super(DataSampler2, self).__init__()

    def sample_dataset(self, dataset, dataloader, net, loss_fn, k, ctx):
        outputs = list()
        for i, batch in enumerate(dataloader):
            d = batch[0].as_in_context(ctx[0])
            output = mx.nd.argmax(net(d)[1], axis=1)
            outputs.append(output)

        outputs = mx.nd.concat(*outputs, dim=0)
        outputs = outputs.asnumpy()
        gt = dataset._label
        err_inds = np.where(outputs != gt)[0]
        rand_inds = npr.permutation(len(err_inds))
        sample_inds = err_inds[rand_inds[:k]]
        if dataset.name == 'ImageNetSplit':
            return Imagenet_SampleDataset(dataset, sample_inds)
        elif dataset.name == 'CIFAR10Split' or 'CIFAR100Split' or 'CORE50Split':
            return SampleDataset(dataset, sample_inds)

# R3 - random sampler
class DataSampler3(DataSampler):
    def __init__(self):
        super(DataSampler3, self).__init__()

    def sample_dataset(self, dataset, dataloader, net, loss_fn, k, ctx):
        rand_inds = npr.permutation(len(dataset._data))
        sample_inds = rand_inds[:k]
        if dataset.name == 'ImageNentSplit':
            return Imagenet_SampleDataset(dataset, sample_inds)
        elif dataset.name == 'CIFAR10Split' or 'CIFAR100Split' or 'CORE50Split':
            return SampleDataset(dataset, sample_inds)

# R4 - range sampler
class DataSampler4(DataSampler):
    def __init__(self):
        super(DataSampler4, self).__init__()

    def sample_dataset(self, dataset, dataloader, net, loss_fn, k, ctx):
        losses = list()
        for i, batch in enumerate(dataloader):
            d = batch[0].as_in_context(ctx[0])
            l = batch[1].as_in_context(ctx[0])
            loss = loss_fn(net(d)[1], l)
            losses.append(loss)

        losses = mx.nd.concat(*losses, dim=0)
        losses = losses.asnumpy()
        sample_inds = np.argsort(losses)[-k*2:-k]
        if dataset.name == 'ImageNetSplit':
            return Imagenet_SampleDataset(dataset, sample_inds)
        elif dataset.name == 'CIFAR10Split' or 'CIFAR100Split' or 'CORE50Split':
            return SampleDataset(dataset, sample_inds)

# R5 - easy example sampler
class DataSampler5(DataSampler):
    def __init__(self):
        super(DataSampler5, self).__init__()

    def sample_dataset(self, dataset, dataloader, net, loss_fn, k, ctx):
        losses = list()
        for i, batch in tqdm(enumerate(dataloader)):
            d = batch[0].as_in_context(ctx[0])
            l = batch[1].as_in_context(ctx[0])
            loss = loss_fn(net(d)[1], l)
            losses.append(loss)
        losses = mx.nd.concat(*losses, dim=0)
        losses = losses.asnumpy()
        sample_inds = np.argsort(losses)[:k]
        if dataset.name == 'ImageNetSplit':
            return Imagenet_SampleDataset(dataset, sample_inds)
        elif dataset.name == 'CIFAR10Split' or 'CIFAR100Split' or 'CORE50Split':
            return SampleDataset(dataset, sample_inds)

# R6 - same distribute
class DataSampler6(DataSampler):
    def __init__(self):
        super(DataSampler6, self).__init__()

    def sample_dataset(self, dataset, dataloader, net, loss_fn, k, ctx):
        losses = list()
        for i, batch in tqdm(enumerate(dataloader)):
            d = batch[0].as_in_context(ctx[0])
            l = batch[1].as_in_context(ctx[0])
            loss = loss_fn(net(d)[1], l)
            losses.append(loss)
        losses = mx.nd.concat(*losses, dim=0)
        losses = losses.asnumpy()
        #count = 5
        select_inds = np.argsort(losses)[5000:9000]
        #count += 1
        sample_inds = random.sample(list(select_inds),k)
        if dataset.name == 'ImageNetSplit':
            return Imagenet_SampleDataset(dataset, np.array(sample_inds))
        elif dataset.name == 'CIFAR10Split' or 'CIFAR100Split' or 'CORE50Split':
            return SampleDataset(dataset, np.array(sample_inds))

# R7 - gaussian distribute
class DataSampler7(DataSampler):
    def __init__(self):
        super(DataSampler7, self).__init__()

    def sample_dataset(self, dataset, dataloader, net, loss_fn, k, ctx):
        losses = list()
        for i, batch in tqdm(enumerate(dataloader)):
            d = batch[0].as_in_context(ctx[0])
            l = batch[1].as_in_context(ctx[0])
            loss = loss_fn(net(d)[1], l)
            losses.append(loss)
        losses = mx.nd.concat(*losses, dim=0)
        losses = losses.asnumpy()
        gs = np.random.normal(8000,2000,k)
        gs = list(set(np.array(gs,int)))
        np_gs = np.array(gs)
        rm_ind = np.where(np.array(gs)>9999)[0]
        rm = np_gs[rm_ind]
        for i in rm:
            gs.remove(i)
        print('use new data %d ...'% len(gs))
        select_inds = np.argsort(losses)[gs]
        if dataset.name == 'ImageNetSplit':
            return Imagenet_SampleDataset(dataset, np.array(select_inds))
        elif dataset.name == 'CIFAR10Split' or 'CIFAR100Split' or 'CORE50Split':
            return SampleDataset(dataset, np.array(select_inds))

# Rtest - test
class DataSamplerTest(DataSampler):
    def __init__(self):
        super(DataSamplerTest, self).__init__()

    def sample_dataset(self, dataset, dataloader, net, loss_fn, k, ctx):
        losses = list()
        outputs = list()
        outs = list()
        t = 0
        wh = list()
        for i, batch in tqdm(enumerate(dataloader)):
            d = batch[0].as_in_context(ctx[0])
            l = batch[1].as_in_context(ctx[0])
            loss = loss_fn(net(d)[1], l)
            losses.append(loss)
            out = net(d)[1]
            output = mx.nd.argmax(out, axis=1)
            outputs.append(output)
            outs.append(out)

        outputs = mx.nd.concat(*outputs, dim=0)
        losses = mx.nd.concat(*losses, dim=0)
        outs = mx.nd.concat(*outs, dim=0)
        outputs = outputs.asnumpy()
        losses = losses.asnumpy()
        gt = dataset._label
        err_inds = np.where(outputs != gt)[0]
        sample_inds = np.argsort(losses)[-k:]
        for i in range(err_inds.shape[0]):
            w = np.where(err_inds[i]==np.argsort(losses))
            wh.append(w[0][0])

        if dataset.name == 'ImageNetSplit':
            return Imagenet_SampleDataset(dataset, sample_inds)
        elif dataset.name == 'CIFAR10Split' or 'CIFAR100Split' or 'CORE50Split':
            return SampleDataset(dataset, sample_inds)

# Pillar sampler
class PillarSampler():
    def __init__(self):
        pass

    def sample_pillarset(self, **kwargs):
        raise Exception("To be implemented in the sub-class.")

#- incremental random sampler

class PillarSampler0(PillarSampler):
    def __init__(self):
        super(PillarSampler0, self).__init__()

    def sample_pillarset(self, pillarset, dataset, dataloader, net, k, ctx):
        rand_inds = npr.permutation(len(dataset._data))
        sample_inds = rand_inds[:k]

        if dataset.name == 'ImageNetSplit':
            return Imagenet_SampleDataset(dataset, np.array(sample_inds))
        elif dataset.name == 'CIFAR10Split' or 'CIFAR100Split' or 'CORE50Split':
            return SampleDataset(dataset, np.array(sample_inds))

#R2 - som sampler
class PillarSampler2(PillarSampler):
    def __init__(self):
        super(PillarSampler2, self).__init__()

    def sample_pillarset(self, pillar_ind, dataset):

        if dataset.name == 'ImageNetSplit':
            return Imagenet_SampleDataset(dataset, np.array(pillar_ind))
        elif dataset.name == 'CIFAR10Split' or 'CIFAR100Split' or 'CORE50Split':
            return SampleDataset(dataset, np.array(pillar_ind))
