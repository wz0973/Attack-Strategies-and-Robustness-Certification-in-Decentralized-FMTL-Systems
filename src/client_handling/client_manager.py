import torch
import torchvision
from client_handling.client import Client
from client_handling.seed import set_seed
from data_handling.cifar import RemappedSubset
from models.fl_detection_model import LandmarkDetectionModel
from models.multilabel_classification_model import MultiLabelClassificationModel
from models.single_label_classification_model import SingleLabelClassificationModel
from attacks.random_label_flip import (
    poison_labels_randomly,
    poison_labels_randomly_for_cifar,
    get_targets_base_dataset_and_indices,
)
from attacks.target_label_flip import (
    poison_labels_targeted,
    poison_labels_targeted_for_cifar,
)
from attacks.trigger_injection import (
    build_mosaic_poison_loader,
    build_cifar_mosaic_poison_loader
)

"""
Client construction utilities for different datasets.
This module handles the injection of data poisoning attacks
and wraps each client with a suitable model and data loader.
"""

def _create_cifar_clients(
    title, t_an, t_ob, dm_cifar, an_client_ids, ob_client_ids, device, backbone_layers,
):
    """
    Creates clients for the CIFAR-10 dataset.

    This includes two task groups:
    - Animal classification (6 classes)
    - Object classification (4 classes)
    Attacks are injected conditionally for AN_Cifar10_C0 based on the experiment title.
    """
    an_cifar_clients = []
    ob_cifar_clients = []

    # Create animal clients
    for idx, id in enumerate(an_client_ids):
        seed = idx + 300
        set_seed(seed)
        train_loader = dm_cifar.train_animals_loaders[idx]
        # ✅ 1. Random label flip attack injection: CIFAR-10 Animals Task
        if id == "AN_Cifar10_C0" and "POISON_Random_Label_Flip" in title:
            print("🎯 Injecting random label flip attack into AN_Cifar10_C0")
            poison_labels_randomly_for_cifar(dm_cifar.train_animals_datasets[idx], num_classes=6, flip_rate=1.0)
        # ✅ 2. Target label flip attack injection: cat ➝ dog
        if id == "AN_Cifar10_C0" and "POISON_Target_Label_Flip" in title:
            print("🎯 Injecting targeted label flip attack into AN_Cifar10_C0 (cat ➝ dog)")
            poison_labels_targeted_for_cifar(dm_cifar.train_animals_datasets[idx], source_label=3, target_label=5,
                                             flip_rate=1.0)
        # ✅ 3. Trigger injection attack injection: Insert mosaic attack
        if id == "AN_Cifar10_C0" and "POISON_Trigger_Injection" in title:
            print("🎯 Injecting mosaic poisoning into AN_Cifar10_C0")
            train_loader = build_cifar_mosaic_poison_loader(dm_cifar.train_animals_datasets[idx])

        an_cifar_clients.append(
            Client(
                c_id=id,
                title=title,
                tasktype=t_an,
                seed=seed,
                model=SingleLabelClassificationModel(
                    num_classes=6,
                    model=torchvision.models.resnet18(pretrained=False),
                    train_loader=train_loader,
                    val_loader=dm_cifar.val_animals_loaders[idx],
                    test_loader=dm_cifar.test_animals_loader,
                    backbone_layers=backbone_layers,
                ).to(device),
            )
        )

    # Create object clients.
    for idx, id in enumerate(ob_client_ids):
        seed = idx + 400
        set_seed(seed)
        ob_cifar_clients.append(
            Client(
                c_id=id,
                title=title,
                tasktype=t_ob,
                seed=seed,
                model=SingleLabelClassificationModel(
                    num_classes=4,
                    model=torchvision.models.resnet18(pretrained=False),
                    train_loader=dm_cifar.train_objects_loaders[idx],
                    val_loader=dm_cifar.val_objects_loaders[idx],
                    test_loader=dm_cifar.test_objects_loader,
                    backbone_layers=backbone_layers,
                ).to(device),
            )
        )

    return an_cifar_clients, ob_cifar_clients


def _create_celeba_clients(
    title,
    t_multilabel,
    t_facial_landmarks,
    dm_celeba,
    ml_client_ids,
    fl_client_ids,
    device,
    backbone_layers,
):
    """
    Creates clients for the CelebA dataset.

    This includes:
    - Multi-label classification task (e.g., smile, glasses, etc.)
    - Facial landmark regression task
    Attack injections apply only to ML_Celeba_C0.
    """
    ml_celeba_clients = []
    fl_celeba_clients = []
    for idx, id in enumerate(ml_client_ids):
        seed = idx + 100
        set_seed(seed)
        train_loader = dm_celeba.train_multilabel_loaders[idx]
        # ✅ 1. Random label flip attack injection: CeleBA Multi-lable classification Task
        if id == "ML_Celeba_C0" and "POISON_Random_Label_Flip" in title:
            print("🎯Injecting label flipping attack into ML_Celeba_C0")
            poison_labels_randomly(dm_celeba.train_multilabel_datasets[idx])
        # ✅ Target label flip attack injection: smiling
        if id == "ML_Celeba_C0" and "POISON_Target_Label_Flip" in title:
            print("🎯 Injecting targeted label flipping attack into ML_Celeba_C0 (Smiling only)")
            poison_labels_targeted(dm_celeba.train_multilabel_datasets[idx], target_label_idx=31, flip_rate=1.0)
        # ✅ 3. Trigger injection attack injection: Insert mosaic attack
        if id == "ML_Celeba_C0" and "POISON_Trigger_Injection" in title:
            print("🎯Enabling on-the-fly mosaic trigger for ML_Celeba_C0")
            train_loader = build_mosaic_poison_loader(dm_celeba.train_multilabel_datasets[idx])

        ml_celeba_clients.append(
            Client(
                c_id=id,
                title=title,
                tasktype=t_multilabel,
                seed=seed,
                model=MultiLabelClassificationModel(
                    num_classes=40,
                    model=torchvision.models.resnet18(pretrained=False),
                    train_loader=train_loader,
                    val_loader=dm_celeba.val_multilabel_loaders[idx],
                    test_loader=dm_celeba.test_multilabel_loader,
                    backbone_layers=backbone_layers,
                ).to(device),
            )
        )

    for idx, id in enumerate(fl_client_ids):
        seed = idx + 200
        set_seed(seed)
        fl_celeba_clients.append(
            Client(
                c_id=id,
                title=title,
                tasktype=t_facial_landmarks,
                seed=seed,
                model=LandmarkDetectionModel(
                    num_landmarks=5,
                    model=torchvision.models.resnet18(pretrained=False),
                    train_loader=dm_celeba.train_landmarks_loaders[idx],
                    val_loader=dm_celeba.val_landmarks_loaders[idx],
                    test_loader=dm_celeba.test_landmarks_loader,
                    backbone_layers=backbone_layers,
                ).to(device),
            )
        )

    return ml_celeba_clients, fl_celeba_clients


def get_clients(
    dataset,
    exp_title,
    task_1,
    task_2,
    data_manager,
    task_1_clients_ids,
    task_2_clients_ids,
    device,
    backbone_layers,
):
    """Decides which clients to create based on the provided dataset. It then calls one of the above two defined functions.

    Args:
        dataset (str): The name of the dataset
        exp_title (str): The title of the experiment
        task_1 (str): The name of the first task group
        task_2 (srt): The name of the second task group
        data_manager (Datamanager): The datamanager
        task_1_clients_ids ([str]): The client ids belonging to the first task group
        task_2_clients_ids ([srt]): The client ids belonging to the second task group
        device (torch.device): The device, gpu or cpu.
        backbone_layers (str): Size of backbone.

    Returns:
        [clients_t1, clients_t2]: The generated clients for each task group.
    """
    print("Getting Clients")
    if dataset == "cifar10":
        return _create_cifar_clients(
            exp_title,
            task_1,
            task_2,
            data_manager,
            task_1_clients_ids,
            task_2_clients_ids,
            device,
            backbone_layers,
        )
    elif dataset == "celeba":
        return _create_celeba_clients(
            exp_title,
            task_1,
            task_2,
            data_manager,
            task_1_clients_ids,
            task_2_clients_ids,
            device,
            backbone_layers,
        )
    else:
        raise "Dataset not properly defined!"