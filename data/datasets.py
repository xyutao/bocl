import os, gzip, tarfile, struct, warnings, numpy as np

from mxnet.gluon.data import dataset
from mxnet.gluon.utils import download, check_sha1, _get_repo_file_url
from mxnet import nd, image, recordio, base
import numpy.random as npr

import pickle as pkl

def merge_datasets(d0, d1):
    return MergeDataset(d0, d1)

def imagenet_merge_datasets(d0, d1):
    return MergeDatasetImageNet(d0, d1)

def brightness_aug(data, brightness):
    return data * (1 + brightness)

def saturation_aug(data, saturation):
    coef = np.array([[[[0.299, 0.587, 0.114]]]])
    gray = data * coef
    gray = np.sum(gray, axis=3, keepdims=True)
    gray *= saturation
    return gray * saturation + data * (1 + saturation)

class CIFAR10Split(dataset._DownloadedDataset):
    def __init__(self, root=os.path.join(base.data_dir(), 'datasets', 'cifar10'),
                 split_id=0, train=True, transform=None):
        self.name='CIFAR10Split'
        self._train = train
        self._split_id = split_id
        self._archive_file = ('cifar-10-binary.tar.gz', 'fab780a1e191a7eda0f345501ccd62d20f7ed891')
        self._train_data = [('data_batch_1.bin', 'aadd24acce27caa71bf4b10992e9e7b2d74c2540'),
                            ('data_batch_2.bin', 'c0ba65cce70568cd57b4e03e9ac8d2a5367c1795'),
                            ('data_batch_3.bin', '1dd00a74ab1d17a6e7d73e185b69dbf31242f295'),
                            ('data_batch_4.bin', 'aab85764eb3584312d3c7f65fd2fd016e36a258e'),
                            ('data_batch_5.bin', '26e2849e66a845b7f1e4614ae70f4889ae604628')]
        self._test_data = [('test_batch.bin', '67eb016db431130d61cd03c7ad570b013799c88c')]
        self._namespace = 'cifar10'
        self._brightness_jitter = [0, -0.1, 0.1, -0.2, 0.2]
        self._saturation_jitter = [0, -0.1, 0.1, -0.2, 0.2]
        super(CIFAR10Split, self).__init__(root, transform)

    def _read_batch(self, filename):
        with open(filename, 'rb') as fin:
            data = np.frombuffer(fin.read(), dtype=np.uint8).reshape(-1, 3072+1)

        data, label = data[:, 1:].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), data[:, 0].astype(np.int32)
        if self._train:
            data = saturation_aug(data, self._saturation_jitter[self._split_id])
            data = brightness_aug(data, self._brightness_jitter[self._split_id])
        return (data, label)

    def _get_data(self):
        if any(not os.path.exists(path) or not check_sha1(path, sha1)
               for path, sha1 in ((os.path.join(self._root, name), sha1)
                                  for name, sha1 in self._train_data + self._test_data)):
            namespace = 'gluon/dataset/'+self._namespace
            filename = download(_get_repo_file_url(namespace, self._archive_file[0]),
                                path=self._root,
                                sha1_hash=self._archive_file[1])

            with tarfile.open(filename) as tar:
                tar.extractall(self._root)

        if self._train:
            data_files = [self._train_data[self._split_id]]
        else:
            data_files = self._test_data
        data, label = zip(*(self._read_batch(os.path.join(self._root, name))
                            for name, _ in data_files))
        data = np.concatenate(data)
        label = np.concatenate(label)
        self._data = nd.array(data, dtype=data.dtype)
        self._label = label

# if the datasets is cifar10,cifar100, core50
class SampleDataset(dataset._DownloadedDataset):
    def __init__(self, dataset, sample_inds, root=os.path.join(base.data_dir(), 'datasets', 'sample'), transform=None):
        super(SampleDataset, self).__init__(root=root, transform=transform)

        self.name=dataset.name
        self._dataset = dataset
        self._sample_inds = sample_inds
        self._data = self._dataset._data[self._sample_inds]
        self._label = self._dataset._label[self._sample_inds]

    def _get_data(self):
        pass


## if the datasets is the imagenet
class Imagenet_SampleDataset(dataset.Dataset):
    def __init__(self, dataset, sample_inds):
        self.name= dataset.name
        self._dataset = dataset
        self._sample_inds = sample_inds
        self.items = []
        for i in self._sample_inds:
            # self.items.extend(self._dataset.items[i])
            self.items.append(self._dataset.items[i])

    def __getitem__(self, idx):
        img = image.imread(self.items[idx][0], 1)
        label = self.items[idx][1]
        return img, label

    def __len__(self):
        return len(self.items)

class MergeDataset(dataset._DownloadedDataset):
    def __init__(self, d0, d1, root=os.path.join(base.data_dir(), 'datasets', 'merge'), transform=None):
        super(MergeDataset, self).__init__(root=root, transform=transform)
        self.name =d0.name
        if d1 is None and d0 is not None:
            self._data, self._label = d0._data, d0._label
        elif d0 is None and d1 is not None:
            self._data, self._label = d1._data, d1._label
        elif d0 is not None and d1 is not None:
            self._data = nd.concat(d0._data, d1._data, dim=0)
            self._label = np.concatenate([d0._label, d1._label])
        else:
            self._data, self._label = None, None

    def _get_data(self):
        pass

class MergeDatasetImageNet(dataset.Dataset):
    def __init__(self, d0, d1):
        self.items = d0.items + d1.items
        self.name = d0.name

    def __getitem__(self, idx):
        img = image.imread(self.items[idx][0], 1)
        label = self.items[idx][1]
        return img, label

    def __len__(self):
        return len(self.items)

global imgs
imgs = np.zeros((1,))

class CORE50Split(dataset._DownloadedDataset):
    def __init__(self, root=os.path.join(base.data_dir(), 'datasets', 'core50'),
                 split_id=0, train=True, transform=None):
        self.name = 'CORE50Split'
        self._train = train
        self._split_id = split_id
        self.globals = globals()
        super(CORE50Split, self).__init__(root, transform)

    def _get_data(self):
        print('loading core50 path ...')
        pkl_file = open('/home/dsl/AAAI20/paths.pkl', 'rb')
        paths = pkl.load(pkl_file)
        label = list()
        test_index = list()
        train_index = list()
        for index, path in enumerate(paths):
            # label.append(int((int(path.split('/')[1][1:])-1)/5))    # classes : 10
            label.append(int(path.split('/')[1][1:]) - 1)  # classes : 50
            test = path.split('/')[0][1:]
            if test == '3' or test == '7' or test == '10':
                test_index.append(index)
            elif test == str(self._split_id):
                train_index.append(index)
        label = np.array(label)

        if self.globals['imgs'].shape[0]==1:
            print('loading core50 dataset ...')
            imgs = np.load('/dataset/dsl/core50_imgs.npz')['x']
            #imgs = imgs.transpose(0, 3, 1, 2)
            self.globals['imgs'] = imgs
        else:
            print('loading core50 dataset from cache ...')
            imgs = self.globals['imgs']

        if self._train:
            data = imgs[train_index]
            label = label[train_index]
        else:
            data = imgs[test_index]
            label = label[test_index]
        self._data = nd.array(data, dtype=data.dtype)
        self._label = label


class CIFAR100Split(dataset._DownloadedDataset):
    def __init__(self, root=os.path.join(base.data_dir(), 'datasets', 'cifar100'),
                 split_id=0, train=True, transform=None):
        self.name = 'CIFAR100Split'
        self._train = train
        self._split_id = split_id
        self._archive_file = ('cifar-100-binary.tar.gz', 'a0bb982c76b83111308126cc779a992fa506b90b')
        self._train_data = [('train.bin', 'e207cd2e05b73b1393c74c7f5e7bea451d63e08e')]
        self._test_data = [('test.bin', '8fb6623e830365ff53cf14adec797474f5478006')]
        self._fine_label = True
        self._namespace = 'cifar100'
        self._brightness_jitter = [0, -0.1, 0.1, -0.2, 0.2]
        self._saturation_jitter = [0, -0.1, 0.1, -0.2, 0.2]

        super(CIFAR100Split, self).__init__(root, transform) # pylint: disable=bad-super-call

    def _read_batch(self, filename):
        with open(filename, 'rb') as fin:
            data = np.frombuffer(fin.read(), dtype=np.uint8).reshape(-1, 3072+2)

        data, label = data[:, 2:].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), \
                      data[:, 0+self._fine_label].astype(np.int32)
        if self._train:
            data = saturation_aug(data, self._saturation_jitter[self._split_id])
            data = brightness_aug(data, self._brightness_jitter[self._split_id])
        return (data, label)

    def _get_data(self):
        if any(not os.path.exists(path) or not check_sha1(path, sha1)
               for path, sha1 in ((os.path.join(self._root, name), sha1)
                                  for name, sha1 in self._train_data + self._test_data)):
            namespace = 'gluon/dataset/'+self._namespace
            filename = download(_get_repo_file_url(namespace, self._archive_file[0]),
                                path=self._root,
                                sha1_hash=self._archive_file[1])

            with tarfile.open(filename) as tar:
                tar.extractall(self._root)

        if self._train:
            data_files = self._train_data
        else:
            data_files = self._test_data
        data, label = zip(*(self._read_batch(os.path.join(self._root, name))
                            for name, _ in data_files))
        data = np.concatenate(data)
        label = np.concatenate(label)
        if self._train:
            npr.seed(0)
            rand_inds = npr.permutation(50000)
            data = data[rand_inds]
            label = label[rand_inds]
            data = data[self._split_id*10000:(self._split_id+1)*10000]
            label = label[self._split_id*10000:(self._split_id+1)*10000]
        self._data = nd.array(data, dtype=data.dtype)
        self._label = label

# from mxnet.gluon.data.vision import CIFAR10
class ImageNetSplit(dataset.Dataset):
    """A dataset for loading image files stored in a folder structure like::

        root/car/0001.jpg
        root/car/xxxa.jpg
        root/car/yyyb.jpg
        root/bus/123.jpg
        root/bus/023.jpg
        root/bus/wwww.jpg

    Parameters
    ----------
    root : str
        Path to root directory.
    flag : {0, 1}, default 1
        If 0, always convert loaded images to greyscale (1 channel).
        If 1, always convert loaded images to colored (3 channels).
    transform : callable, default None
        A function that takes data and label and transforms them:
    ::

        transform = lambda data, label: (data.astype(np.float32)/255, label)

    Attributes
    ----------
    synsets : list
        List of class names. `synsets[i]` is the name for the integer label `i`
    items : list of tuples
        List of all images in (filename, label) pairs.
    """

    def __init__(self, root='/dataset/imagenet/', flag=1, split_id=0, train=True, transform=None, method='naive'):
        self.name = 'ImageNetSplit'
        self._split_id = split_id
        self._root = os.path.expanduser(root)
        self._method = method
        if train:
            self._root = os.path.join(self._root, 'ILSVRC2012_img_train/')
        else:
            self._root = os.path.join(self._root, 'ILSVRC2012_img_val/')
        self._flag = flag
        self._transform = transform
        self._exts = ['.jpg', '.jpeg', '.png']
        if train:
            self._list_images_train(self._root, self._split_id)
        else:
            self._list_images_val(self._root)

    def _list_images_train(self, root, split_id):
        self.synsets = []
        self.items = []
        self.items_all = []

        for folder in sorted(os.listdir(root)):
            path = os.path.join(root, folder)
            if not os.path.isdir(path):
                warnings.warn('Ignoring %s, which is not a directory.' % path, stacklevel=3)
                continue
            label = len(self.synsets)
            self.synsets.append(folder)

            l = 0
            npc = 250 # num per class for all session
            for filename in sorted(os.listdir(path)):
                l += 1
                filename = os.path.join(path, filename)
                ext = os.path.splitext(filename)[1]
                if ext.lower() not in self._exts:
                    warnings.warn('Ignoring %s of type %s. Only support %s' % (
                        filename, ext, ', '.join(self._exts)))
                    continue
                self.items_all.append((filename, label))
                if l == npc:
                    break

        if self._method == 'full_naive' or self._method == 'full_pillar_all':
            for i in range(1000):
                self.items.extend(self.items_all[(npc * i + 50 * split_id):(npc * i + 50 * (split_id + 1))])

        elif self._method == 'full_pillar':
            if split_id==0:
                for i in range(1000):
                    self.items.extend(self.items_all[(npc * i + 50 * split_id):(npc * i + 50 * (split_id + 1))])
            else:
                for i in range(1000):
                    self.items.extend(self.items_all[(npc * i + 50 * split_id - 10):(npc * i + 50 * (split_id + 1))])
        elif self._method == 'full_cum':
            if split_id==0:
                for i in range(1000):
                    self.items.extend(self.items_all[(npc * i):(npc * (i+1))])
            else:
                raise EOFError('full_cum split_id is not equal 0')

        elif self._method == 'rand_naive' or self._method == 'rand_pillar_all':
            if split_id==0:
                for i in range(1000):
                    self.items.extend(self.items_all[(npc * i + 50 * split_id):(npc * i + 50 * (split_id + 1))])
            else:
                for i in range(1000):
                    self.items.extend(self.items_all[(npc * i + 50 + 10*(split_id-1)):(npc * i + 50 + 10*split_id)])
        elif self._method == 'rand_pillar':
            if split_id==0:
                for i in range(1000):
                    self.items.extend(self.items_all[(npc * i + 50 * split_id):(npc * i + 50 * (split_id + 1))])
            else:
                for i in range(1000):
                    self.items.extend(self.items_all[(npc * i + 50 + 10*(split_id-2)):(npc * i + 50 + 10*split_id)])
        elif self._method == 'rand_cum':
            if split_id==0:
                for i in range(1000):
                    self.items.extend(self.items_all[(npc * i):(npc * i + 90)])
            else:
                raise EOFError('full_cum split_id is not equal 0')


    def _list_images_val(self, root):
        self.synsets = []
        self.items = []

        for folder in sorted(os.listdir(root)):
            path = os.path.join(root, folder)
            if not os.path.isdir(path):
                warnings.warn('Ignoring %s, which is not a directory.' % path, stacklevel=3)
                continue
            label = len(self.synsets)
            self.synsets.append(folder)
            for filename in sorted(os.listdir(path)):
                filename = os.path.join(path, filename)
                ext = os.path.splitext(filename)[1]
                if ext.lower() not in self._exts:
                    warnings.warn('Ignoring %s of type %s. Only support %s' % (
                        filename, ext, ', '.join(self._exts)))
                    continue
                self.items.append((filename, label))

    def __getitem__(self, idx):
        img = image.imread(self.items[idx][0], self._flag)
        label = self.items[idx][1]
        if self._transform is not None:
            return self._transform(img, label)
        return img, label

    def __len__(self):
        return len(self.items)
