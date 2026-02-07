import logging
import random

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os

from .datasets import MNIST_truncated

from data_preprocessing.utils.imbalance_data import ImbalancedDatasetSampler
from utils.data_partition_io import save_partition_once, wait_and_load_partition


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of non-unique client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    clients = sorted(cdata['users'])

    return clients, groups, train_data, test_data


def batch_data(data, batch_size):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    batch_data = list()
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i + batch_size]
        batched_y = data_y[i:i + batch_size]
        batched_x = torch.from_numpy(np.asarray(batched_x)).float()
        batched_y = torch.from_numpy(np.asarray(batched_y)).long()
        batch_data.append((batched_x, batched_y))
    return batch_data


def load_partition_data_mnist_by_device_id(batch_size,
                                           device_id,
                                           train_path="MNIST_mobile",
                                           test_path="MNIST_mobile"):
    train_path += '/' + device_id + '/' + 'train'
    test_path += '/' + device_id + '/' + 'test'
    return load_partition_data_mnist(batch_size, train_path, test_path)



def _data_transforms_mnist():
    MNIST_MEAN = (0.1307,)
    MNIST_STD = (0.3081,)

    image_size = 28
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=MNIST_MEAN , std=MNIST_STD),
        ])
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=MNIST_MEAN , std=MNIST_STD),
        ])

    return train_transform, test_transform



def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts


def load_mnist_data(datadir):
    train_transform, test_transform = _data_transforms_mnist()

    mnist_train_ds = MNIST_truncated(datadir, train=True, download=True, transform=train_transform)
    mnist_test_ds = MNIST_truncated(datadir, train=False, download=True, transform=test_transform)

    X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
    X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

    return (X_train, y_train, X_test, y_test)



def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, args=None, download=False):
    return get_dataloader_MNIST(datadir, train_bs, test_bs, dataidxs, args=args, download=download)


def get_dataloader_MNIST(datadir, train_bs, test_bs, dataidxs=None, args=None, download=False):
    dl_obj = MNIST_truncated
    train_tf, test_tf = _data_transforms_mnist()
    train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True,  transform=train_tf,  download=download)
    test_ds  = dl_obj(datadir, train=False,                transform=test_tf, download=download)


    sampler_mode = getattr(args, "data_sampler", None) if args is not None else None
    if sampler_mode in ["imbalance", "decay_imb"]:
        train_sampler = ImbalancedDatasetSampler(args, train_ds, class_num=10)
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=(train_sampler is None),
                                       num_workers=0,          # ①
                                        pin_memory=False, 
                                   drop_last=True, sampler=train_sampler)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, 
                                    num_workers=0,          # ①
                                    pin_memory=False, 
                                  drop_last=True)
    else:
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, 
    num_workers=0,          # ①
    pin_memory=False, 
    drop_last=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, 
    num_workers=0,          # ①
    pin_memory=False, 
    drop_last=True)

    return train_dl, test_dl



def partition_data(dataset, datadir, partition, n_nets, alpha):
    logging.info("*********partition data***************")
    X_train, y_train, X_test, y_test = load_mnist_data(datadir)
    n_train = X_train.shape[0]
    # n_test = X_test.shape[0]

    if partition == "homo":
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero":
        min_size = 0
        K = 10
        N = y_train.shape[0]
        logging.info("N = " + str(N) + str(alpha))

        net_dataidx_map = {}

        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    # refer to https://github.com/Xtra-Computing/NIID-Bench/blob/main/utils.py
    elif partition > "noniid-#label0" and partition <= "noniid-#label9":
        num = eval(partition[13:])
        if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
            num = 1
            K = 2
        else:
            K = 10
        if num == 10:
            net_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(n_nets)}
            for i in range(10):
                idx_k = np.where(y_train==i)[0]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k, n_nets)
                for j in range(n_nets):
                    net_dataidx_map[j]=np.append(net_dataidx_map[j],split[j])
        else:
            times=[0 for i in range(10)]
            contain=[]
            for i in range(n_nets):
                current=[i%K]
                times[i%K]+=1
                j=1
                while (j<num):
                    ind=random.randint(0,K-1)
                    if (ind not in current):
                        j=j+1
                        current.append(ind)
                        times[ind]+=1
                contain.append(current)
            net_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(n_nets)}
            for i in range(K):
                idx_k = np.where(y_train==i)[0]
                np.random.shuffle(idx_k)
                if times[i] == 0:   # fix
                    continue
                split = np.array_split(idx_k, times[i])
                ids=0
                for j in range(n_nets):
                    if i in contain[j]:
                        net_dataidx_map[j]=np.append(net_dataidx_map[j],split[ids])
                        ids+=1

            # for i in range(K):
            #     idx_k = np.where(y_train==i)[0]
            #     np.random.shuffle(idx_k)
            #     split = np.array_split(idx_k,times[i])
            #     ids=0
            #     for j in range(n_nets):
            #         if i in contain[j]:
            #             net_dataidx_map[j]=np.append(net_dataidx_map[j],split[ids])
            #             ids+=1
    elif partition == "long-tail":
        if n_nets == 10 or n_nets == 100:
            pass
        else:
            raise NotImplementedError
        
        # There are  n_nets // 10 clients share the \alpha proportion of data of one class
        main_prop = alpha / (n_nets // 10)

        # There are (n_nets - n_nets // 10) clients share the tail of one class
        tail_prop = (1 - main_prop) / (n_nets - n_nets // 10)

        net_dataidx_map = {}
        # for each class in the dataset
        K = 10
        idx_batch = [[] for _ in range(n_nets)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.array([ tail_prop for _ in range(n_nets)])
            main_clients = np.array([ k + i*K for i in range(n_nets // K)])
            proportions[main_clients] = main_prop
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    if partition == "hetero-fix":
        pass
        # distribution_file_path = './data_preprocessing/non-iid-distribution/CIFAR10/distribution.txt'
        # traindata_cls_counts = read_data_distribution(distribution_file_path)
    else:
        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts


def load_partition_data_mnist(dataset, data_dir, partition_method, partition_alpha,
                              client_number, batch_size, args=None):
    

    rank = _get_rank(args)
    # NOTE: comment translated from Chinese
    from pathlib import Path

    current_dir = Path(__file__).parent
    part_path = current_dir / "partitions" / str(client_number) / "mnist_part.json"

    if part_path.exists() and part_path.stat().st_size > 0:
        pass
    else:
        if rank == 0:
            # NOTE: comment translated from Chinese
            X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
                dataset, data_dir, partition_method, client_number, partition_alpha
            )
            # NOTE: comment translated from Chinese
            serializable = {str(k): list(map(int, v)) for k, v in net_dataidx_map.items()}
            save_partition_once(part_path, serializable)

    # NOTE: comment translated from Chinese
    part = wait_and_load_partition(part_path)
    dataidxs = np.array(part[str(rank)], dtype=np.int64)

    np.random.shuffle(dataidxs)
    split = int(len(dataidxs) * 0.7)
    train_idxs, test_idxs = dataidxs[:split], dataidxs[split:]
    download_flag = (rank == 0)
    if hasattr(args, "download_once") and args.download_once is False:
        download_flag = False

    train_dl, _ = get_dataloader(dataset, data_dir, batch_size, batch_size, train_idxs,
                             args=args, download=download_flag)
    _, test_dl  = get_dataloader(dataset, data_dir, batch_size, batch_size, test_idxs,
                             args=args, download=download_flag)

    train_data_num = len(train_idxs)
    test_data_num = len(test_idxs)
    data_local_num_dict = {rank: train_data_num}
    train_data_local_dict = {rank: train_dl}
    test_data_local_dict = {rank: test_dl}
    class_num = 10

    # NOTE: comment translated from Chinese
    train_data_global = None
    test_data_global = None

    return (train_data_num, test_data_num, train_data_global, test_data_global,
            data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num)



def _get_rank(args=None, default=0):
    # NOTE: comment translated from Chinese
    if args is not None and hasattr(args, "rank"):
        return int(args.rank)
    # NOTE: comment translated from Chinese
    for k in ("RANK", "OMPI_COMM_WORLD_RANK", "MV2_COMM_WORLD_RANK"):
        v = os.environ.get(k)
        if v is not None and v.isdigit():
            return int(v)
    # NOTE: comment translated from Chinese
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return default








