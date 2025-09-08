from matplotlib import pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
from torch.utils.data import random_split

"""
Datamanager to handle the celeba dataset. Also includes a RemappedSubset class.
"""


# This is a helper class for the main Cifar Datamanager, defined further below. It is needed as is needed
# since the class labels of the dataset are filtered into two subsets, Animals and Objects.
class RemappedSubset(Dataset):
    def __init__(self, subset, class_map, transform):
        """Initializes the RemappedSubset with a given subset, class map, and transform.

        Args:
            subset (Dataset): The original dataset or subset to wrap and filter.
            class_map (dict): A dictionary to map original class labels to new class labels.
            transform (callable): A function to apply transformations to the input images.
        """
        self.subset = subset
        self.class_map = class_map
        self.transform = transform

    def __len__(self):
        """Returns the number of samples in the subset.

        Returns:
            int: Number of samples in the subset.
        """
        return len(self.subset)

    def __getitem__(self, idx):
        """Retrieves a sample from the subset and applies class remapping.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Returns a tuple: (transformed_image, remapped_label) for the given index.
        """
        img, label = self.subset[idx]
        # ✅ 防止标签已被 poison 后再次 remap 报错
        if label in self.class_map:
            label = self.class_map[label]
        return img, label

class DMCifar:
    def __init__(self, seed, an_num_clients, ob_num_clients, dataset_fraction):
        """Initializes the Datamanager. the data manager is specific to the cifar10 dataset.
        It has train, val and test loaders for each client. The loaders are different for each client.
        There exists no overlap between train and validation data among clients.
        Test data is the same among clients belonging to the same task.

        Args:
            seed (int): The seed
            an_num_clients (int): The number of clients classifying Animals.
            ob_num_clients (int): The number of clients classifying Objects.
            dataset_split ([x,x]]): How the data shall be split among task. For instance [0.5, 0.5] results in an equal split.
        """
        np.random.seed(seed)
        self.an_num_clients = an_num_clients
        self.ob_num_clients = ob_num_clients
        self.dataset_fraction = dataset_fraction
        (
            self.train_animals_datasets,
            self.val_animals_datasets,
            self.test_animals_dataset,
            self.train_objects_datasets,
            self.val_objects_datasets,
            self.test_objects_dataset,
        ) = self._prepare_dataset_splits()
        (
            self.train_animals_loaders,
            self.val_animals_loaders,
            self.test_animals_loader,
            self.train_objects_loaders,
            self.val_objects_loaders,
            self.test_objects_loader,
        ) = self._prepare_loaders()

    def _prepare_dataset_splits(self):
        """Loads the data and splits it into train, validation and test data for each client.

        Returns:
            list of datasets per task: Returns the train_animals_datasets, val_animals_datasets,
            test_animals, train_objects_datasets, val_objects_datasets and test_objects
        """

        # Define Transformation with the aim to reduce overfitting slightly.
        # Normalization from here: https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/cifar10-baseline.html
        cifar10_normalization = torchvision.transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
        )
        train_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomRotation(10),
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
                ),  # also applied a slight random rotation and color jitter
                torchvision.transforms.ToTensor(),
                cifar10_normalization,
            ]
        )
        test_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                cifar10_normalization,
            ]
        )

        # Load CIFAR-10 dataset
        full_train_cifar10 = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=train_transforms
        )
        full_test_cifar10 = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=test_transforms
        )

        train_classes = full_train_cifar10.classes
        test_classes = full_test_cifar10.classes

        # Animal classes and their corresponding indices
        animal_classes = ["bird", "cat", "deer", "dog", "frog", "horse"]
        train_animal_indices = [train_classes.index(cls) for cls in animal_classes]
        test_animal_indices = [test_classes.index(cls) for cls in animal_classes]

        # Object classes and their corresponding indices
        object_classes = ["airplane", "automobile", "ship", "truck"]
        train_object_indices = [train_classes.index(cls) for cls in object_classes]
        test_object_indices = [test_classes.index(cls) for cls in object_classes]

        # Filter datasets by class with label remapping for both animal and object clients
        train_animals = self._filter_dataset_by_class(
            full_train_cifar10,
            train_animal_indices,
            remap_labels=True,
            transforms=train_transforms,
        )
        test_animals = self._filter_dataset_by_class(
            full_test_cifar10,
            test_animal_indices,
            remap_labels=True,
            transforms=test_transforms,
        )

        train_objects = self._filter_dataset_by_class(
            full_train_cifar10,
            train_object_indices,
            remap_labels=True,
            transforms=train_transforms,
        )
        test_objects = self._filter_dataset_by_class(
            full_test_cifar10,
            test_object_indices,
            remap_labels=True,
            transforms=test_transforms,
        )
        
        # Reduce the dataset size by a fraction. if dataset_fraction = 1, the full dataset is processed.
        prev_lenght_of_train_animals = len(train_animals)
        prev_lenght_of_train_objects = len(train_objects)
        train_animals, _ = random_split(train_animals, [int(self.dataset_fraction * len(train_animals)), int((1-self.dataset_fraction) * len(train_animals))])
        train_objects, _ = random_split(train_objects, [int(self.dataset_fraction * len(train_objects)), int((1-self.dataset_fraction) * len(train_objects))])
        if self.dataset_fraction == 1:
            print("Checking dataset size")
            assert prev_lenght_of_train_animals == len(train_animals)
            assert prev_lenght_of_train_objects == len(train_objects)
            print("Dataset size OK")

        # Split datasets into training and validation sets, then further split for clients
        train_animals_datasets, val_animals_datasets = self._split_dataset(
            train_animals, self.an_num_clients, val_split=0.2
        )
        train_objects_datasets, val_objects_datasets = self._split_dataset(
            train_objects, self.ob_num_clients, val_split=0.2
        )

        return (
            train_animals_datasets,
            val_animals_datasets,
            test_animals,
            train_objects_datasets,
            val_objects_datasets,
            test_objects,
        )

    def _filter_dataset_by_class(
        self, dataset, class_indices, remap_labels=False, transforms=None
    ):
        """Filters a given dataset to only include samples that belong to specific classes.
        Optionally remaps the labels of these samples, and returns a subset of the original dataset.

        Args:
            dataset (_type_): _description_
            class_indices (_type_): The indices of the classes that shall be filtered.
            remap_labels (bool, optional): Whether or not the labels shall be ramapped. Defaults to False.
            transforms (_type_, optional): The type of transform that shall be applied. Defaults to None.

        Returns:
            RemappedSubset: The filtered Dataset
        """

        indices = [i for i, (_, label) in enumerate(dataset) if label in class_indices]
        subset = Subset(dataset, indices)

        if remap_labels:
            class_map = {original: idx for idx, original in enumerate(class_indices)}
            subset = RemappedSubset(subset, class_map, transforms)

        return subset

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
            List of all dataloaders for each client: train_animals_loaders, val_animals_loaders,
            test_animals_loader, train_objects_loaders, val_objects_loaders and test_objects_loader.
        """

        # Define the batch size
        batch_size = 64

        train_animals_loaders = [
            DataLoader(ds, batch_size=batch_size, shuffle=True)
            for ds in self.train_animals_datasets
        ]
        train_objects_loaders = [
            DataLoader(ds, batch_size=batch_size, shuffle=True)
            for ds in self.train_objects_datasets
        ]

        val_animals_loaders = [
            DataLoader(ds, batch_size=batch_size, shuffle=False)
            for ds in self.val_animals_datasets
        ]
        val_objects_loaders = [
            DataLoader(ds, batch_size=batch_size, shuffle=False)
            for ds in self.val_objects_datasets
        ]

        test_animals_loader = DataLoader(
            self.test_animals_dataset, batch_size=batch_size, shuffle=False
        )
        test_objects_loader = DataLoader(
            self.test_objects_dataset, batch_size=batch_size, shuffle=False
        )

        return (
            train_animals_loaders,
            val_animals_loaders,
            test_animals_loader,
            train_objects_loaders,
            val_objects_loaders,
            test_objects_loader,
        )
