import torchvision
import os
import random
import numpy as np
import tqdm
import pickle
import torch.utils.data as data


def load_mnist(dataset_path:str):
    # load MNIST dataset
    train_mnist_dataset = torchvision.datasets.MNIST(root=dataset_path, train=True, download=True)
    val_mnist_dataset = torchvision.datasets.MNIST(root=dataset_path, train=False, download=True)

    return train_mnist_dataset, val_mnist_dataset


def dataset_filte_by_class_list(
    dataset:data.Dataset, 
    target_class_lists: list[list[int]]):

    sample_idx_lists = []

    for target_class_list in target_class_lists:
        
        sample_idx_lists.append([idx for idx in range(len(dataset)) if dataset[idx][1] in target_class_list])

    return sample_idx_lists

def select_samples(
    dataset:data.Dataset, 
    num_samples: int,
    c: list[list[int]],
    random_seed: int=0):
    data_x, data_y = [],[]

    request_sample = num_samples * 4



    with  tqdm.tqdm(total=num_samples) as pbar:
        while len(data_x) < num_samples:
            idx = random.randint(0, len(dataset) - 1)
            img, label = dataset[idx]
            if label not in data_y:
                data_x.append(img)
                data_y.append(label)
                pbar.update(1)
    
    return data_x, data_y

def generate_pvr_datasets(
    dataset_path:str, 
    num_samples:list[int]=[10000,1000],
    random_seed:int=0,
    dataset_name:str="pvr_dataset"
    ):

    # Check if dataset path exists
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    train_dataset, val_dataset = load_mnist(dataset_path)

    pvr_dataset_path = os.path.join(dataset_path, dataset_name)

    c = [
        [1,2,3,4,5,6,7,8,9,0],
        [1,2,3,4,5,6,7,8,9,0],
        [1,2,3,4,5,6,7,8,9,0],
        [1,2,3,4,5,6,7,8,9,0]
    ]

    # generate PVR dataset
    train_x, train_y = select_samples(
        train_dataset, 
        num_samples[0], 
        c, 
        random_seed)
    val_x, val_y = select_samples(
        val_dataset, 
        num_samples[1], 
        c, 
        random_seed)

    print(f"Train Dataset Shapes: {np.array(train_x).shape}, {np.array(train_y).shape}")
    print(f"Val Dataset Shapes: {np.array(val_x).shape}, {np.array(val_y).shape}")

    # save PVR dataset
    with open(os.path.join(pvr_dataset_path, f'train_dataset.pkl'), 'wb') as f:
        pickle.dump((train_x, train_y), f)
    with open(os.path.join(pvr_dataset_path, f'val_dataset.pkl'), 'wb') as f:
        pickle.dump((val_x, val_y), f)