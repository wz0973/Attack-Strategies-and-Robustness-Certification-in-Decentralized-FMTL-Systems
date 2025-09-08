from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Subset
import torch
import torchvision
import pandas as pd
from torchvision.datasets import CelebA
from torch.utils.data import random_split

"""
Datamanager to handle the celeba dataset.
"""


class DMCelebA:
    def __init__(self, seed, ml_num_clients, fl_num_clients, dataset_fraction):
        """Initializes the Data manager. the data manager is specific to the celeba dataset.
        It has train, val and test loaders for each client. The loaders are different for each client.
        There exists no overlap between train and validation data among clients.
        Test data is the same among clients belonging to the same task.

        Args:
            seed (int): The seed
            ml_num_clients (int): The number of clients belonging to the multi-label classifation task.
            fl_num_clients (int): The number of clients belonging to the facial landmark detection task.
            dataset_split ([x,x]]): How the data shall be split among task. For instance [0.5, 0.5] results in an equal split.
        """
        np.random.seed(seed)
        self.ml_num_clients = ml_num_clients
        self.fl_num_clients = fl_num_clients
        self.dataset_fraction = dataset_fraction
        self.dataset_split = [0.5, 0.5]
        (
            self.train_multilabel_datasets,
            self.val_multilabel_datasets,
            self.test_multilabel_dataset,
            self.train_landmarks_datasets,
            self.val_landmarks_datasets,
            self.test_landmarks_dataset,
        ) = self._prepare_dataset_splits()
        (
            self.train_multilabel_loaders,
            self.val_multilabel_loaders,
            self.test_multilabel_loader,
            self.train_landmarks_loaders,
            self.val_landmarks_loaders,
            self.test_landmarks_loader,
        ) = self._prepare_loaders()

    def _prepare_dataset_splits(self):
        """Loads the data and splits it into train, validation and test data for each client.

        Returns:
            list of datasets per task: Returns the train_multilabel_datasets, val_multilabel_datasets, test_multilabel_dataset, train_landmark_datasets,
            val_landmark_datasets and test_landmark_dataset.
        """

        # Define Transformation with the aim to reduce overfitting slightly.
        train_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomRotation(10),
                torchvision.transforms.ToTensor(),
            ]
        )
        test_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
            ]
        )

        # Load CelebA dataset with landmark annotations
        full_train_multilabel_dataset = CelebA(
            root="./data", split="train", transform=train_transforms, download=True
        )
        test_multilabel_dataset = CelebA(
            root="./data", split="test", transform=test_transforms, download=True
        )
        full_train_landmark_dataset = CelebA(
            root="./data",
            split="train",
            target_type="landmarks",
            transform=train_transforms,
            download=True,
        )
        test_landmark_dataset = CelebA(
            root="./data",
            split="test",
            target_type="landmarks",
            transform=test_transforms,
            download=True,
        )

        # Convert landmarks to float (usually already float, but ensure consistency)
        full_train_multilabel_dataset.attr = full_train_multilabel_dataset.attr.float()
        test_multilabel_dataset.attr = test_multilabel_dataset.attr.float()
        full_train_landmark_dataset.landmarks_align = (
            full_train_landmark_dataset.landmarks_align.float()
        )
        test_landmark_dataset.landmarks_align = (
            test_landmark_dataset.landmarks_align.float()
        )

        # Get the indexes in a random order
        all_indices = torch.randperm(
            len(full_train_multilabel_dataset)
        )  # both have the same indices

        # Ensure that there is no overlap in data samples between the two tasks.
        train_multilabel_size = int(
            self.dataset_split[0] * len(full_train_multilabel_dataset) * self.dataset_fraction # also take the dataset fraction into account
        )
        train_landmark_size = int(
            self.dataset_split[1] * len(full_train_landmark_dataset) * self.dataset_fraction # also take the dataset fraction into account
        )

        train_multilabel_indices = all_indices[:train_multilabel_size]
        train_landmark_indices = all_indices[
            train_multilabel_size : train_multilabel_size + train_landmark_size
        ]

        multilabel_dataset = Subset(
            full_train_multilabel_dataset, train_multilabel_indices
        )
        landmark_dataset = Subset(
            full_train_landmark_dataset, train_landmark_indices
        )

        # Split the train data into train and validation data for each client.
        train_multilabel_datasets, val_multilabel_datasets = self._split_dataset(
            multilabel_dataset, self.ml_num_clients, val_split=0.2
        )
        train_landmark_datasets, val_landmark_datasets = self._split_dataset(
            landmark_dataset, self.fl_num_clients, val_split=0.2
        )

        # Return all the datasets. Each client has its own traina and val dataset.
        # Clients of the same taskgroup will share the same test set.
        return (
            train_multilabel_datasets,
            val_multilabel_datasets,
            test_multilabel_dataset,
            train_landmark_datasets,
            val_landmark_datasets,
            test_landmark_dataset,
        )

    def _plot_distributions(self):
        """A function to plot the distribution of the celeba dataset labels.

        Returns:
            Figure: A plot.
        """

        # Extract attributes for each dataset
        def extract_attributes_multilabel(subset):
            # Access the original dataset
            dataset = subset.dataset

            # Retrieve the subset indices
            subset_indices = subset.indices

            if subset_indices is not None:
                attributes = dataset.attr[subset_indices]
            else:
                attributes = dataset.attr
            attr_names = dataset.attr_names[: attributes.shape[1]]
            attributes_df = pd.DataFrame(attributes.numpy(), columns=attr_names)
            return attributes_df.sum()

        for i, multilabel_sub_dataset in enumerate(self.train_multilabel_datasets):
            attributes = extract_attributes_multilabel(multilabel_sub_dataset)

            plt.subplot(int(len(self.train_multilabel_datasets) / 2) + 1, 2, i + 1)
            attributes.plot(kind="bar")
            plt.title(
                f"Train Dataset Multilabel - (Samples: {len(multilabel_sub_dataset)})"
            )
            plt.xlabel("Attributes")
            plt.ylabel("Count")
            plt.xticks(rotation=90)

        plt.tight_layout()
        plt.show()

    def _split_dataset(self, dataset, num_splits, val_split=0.2):
        """Splits the the passed train dataset into train and validation for each client.

        Args:
            dataset (dataset): the train dataset
            num_splits (int): The amount of splits that shall be generated. Corresponds to the amount of clients of this task.
            val_split (float, optional): How much of the train data shall be transferred into validaiton data. Defaults to 0.2.

        Returns:
            Dataset: Train and validation data for each client of the task.
        """
        # First split the dataset into training and validation sets
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Split the training dataset into client-specific subsets
        train_subset_size = train_size // num_splits
        train_remainder = train_size % num_splits
        train_lengths = [
            train_subset_size + 1 if i < train_remainder else train_subset_size
            for i in range(num_splits)
        ]
        train_subsets = random_split(train_dataset, train_lengths)

        # Split the validation dataset into client-specific subsets
        val_subset_size = val_size // num_splits
        val_remainder = val_size % num_splits
        val_lengths = [
            val_subset_size + 1 if i < val_remainder else val_subset_size
            for i in range(num_splits)
        ]
        val_subsets = random_split(val_dataset, val_lengths)

        return train_subsets, val_subsets

    def _prepare_loaders(self):
        """Prepares the data loaders

        Returns:
            List of all dataloaders for each client: train_multilabel_loaders, val_multilabel_loaders, test_multilabel_loader,
            train_landmarks_loaders, val_landmarks_loaders, test_landmarks_loader,
        """

        # Define the Batch Size
        batch_size = 64

        train_multilabel_loaders = [
            DataLoader(ds, batch_size=batch_size, shuffle=True)
            for ds in self.train_multilabel_datasets
        ]
        train_landmarks_loaders = [
            DataLoader(ds, batch_size=batch_size, shuffle=True)
            for ds in self.train_landmarks_datasets
        ]

        val_multilabel_loaders = [
            DataLoader(ds, batch_size=batch_size, shuffle=False)
            for ds in self.val_multilabel_datasets
        ]
        val_landmarks_loaders = [
            DataLoader(ds, batch_size=batch_size, shuffle=False)
            for ds in self.val_landmarks_datasets
        ]

        test_multilabel_loader = DataLoader(
            self.test_multilabel_dataset, batch_size=batch_size, shuffle=False
        )
        test_landmarks_loader = DataLoader(
            self.test_landmarks_dataset, batch_size=batch_size, shuffle=False
        )

        return (
            train_multilabel_loaders,
            val_multilabel_loaders,
            test_multilabel_loader,
            train_landmarks_loaders,
            val_landmarks_loaders,
            test_landmarks_loader,
        )
