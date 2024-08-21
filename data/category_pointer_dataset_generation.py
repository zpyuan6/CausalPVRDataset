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

    for index, target_class_list in enumerate(target_class_lists) :
        sample_idx_lists.append([idx for idx in range(dataset.__len__()) if dataset.__getitem__(idx)[1] in target_class_list])

    return sample_idx_lists

def select_samples(
    dataset:data.Dataset, 
    num_samples: int,
    class_index_lists: list[list[int]]):

    data_x, data_y = [],[]

    availible_sample_idx_lists = dataset_filte_by_class_list(dataset, class_index_lists)

    selected_samples_idx_lists = [random.choices(availible_sample_idx_lists[i], k=num_samples) for i in range(len(availible_sample_idx_lists))]

    with  tqdm.tqdm(total=num_samples, desc="Constructure Data Sample") as pbar:
        for idx in range(num_samples):

            labels = []
            
            top_img = np.hstack((dataset.__getitem__(selected_samples_idx_lists[0][idx])[0],dataset.__getitem__(selected_samples_idx_lists[1][idx])[0]))
            labels.append(dataset.__getitem__(selected_samples_idx_lists[0][idx])[1])
            labels.append(dataset.__getitem__(selected_samples_idx_lists[1][idx])[1])

            bottom_img = np.hstack((dataset.__getitem__(selected_samples_idx_lists[2][idx])[0],dataset.__getitem__(selected_samples_idx_lists[3][idx])[0]))
            labels.append(dataset.__getitem__(selected_samples_idx_lists[2][idx])[1])
            labels.append(dataset.__getitem__(selected_samples_idx_lists[3][idx])[1])

            whole_image = np.vstack((top_img,bottom_img))
            input_image = np.expand_dims(whole_image, axis=0)

            data_x.append(input_image)
            data_y.append(labels)

            pbar.update(1)
    
    return data_x, data_y

def generate_pvr_datasets(
    dataset_path:str, 
    num_samples:list[int]=[50000,5000],
    random_seed:int=None,
    dataset_name:str="pvr_dataset",
    class_index_lists: list[list[int]] = [
        [1,2,3,4,5,6,7,8,9,0],
        [1,2,3,4,5,6,7,8,9,0],
        [1,2,3,4,5,6,7,8,9,0],
        [1,2,3,4,5,6,7,8,9,0]
        ]
    ):

    # Check if dataset path exists
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    # Set random seed
    if random_seed!=None:
        random.seed(random_seed)

    train_dataset, val_dataset = load_mnist(dataset_path)

    pvr_dataset_path = os.path.join(dataset_path, dataset_name)

    if not os.path.exists(pvr_dataset_path):
        os.makedirs(pvr_dataset_path)

    # generate PVR dataset

    train_x, train_y = select_samples(
        train_dataset, 
        num_samples[0], 
        class_index_lists)
    val_x, val_y = select_samples(
        val_dataset, 
        num_samples[1], 
        class_index_lists)

    print(f"Train Dataset Shapes: {np.array(train_x).shape}, {np.array(train_y).shape}")
    print(f"Val Dataset Shapes: {np.array(val_x).shape}, {np.array(val_y).shape}")

    # save PVR dataset
    with open(os.path.join(pvr_dataset_path, f'train_dataset.pkl'), 'wb') as f:
        pickle.dump((train_x, train_y), f)
    with open(os.path.join(pvr_dataset_path, f'val_dataset.pkl'), 'wb') as f:
        pickle.dump((val_x, val_y), f)