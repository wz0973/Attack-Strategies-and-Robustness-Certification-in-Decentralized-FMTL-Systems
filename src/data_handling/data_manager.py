from data_handling.celeba import DMCelebA
from data_handling.cifar import DMCifar

"""
Helper script to streamline the retrieval of datamanagers.
"""


def get_data_manager(dataset, task_1_clients_ids, task_2_clients_ids, dataset_fraction):
    """
    Intializes and returns a dataset specific data manager.

    Args:
        dataset (str): The name of the dataset
        task_1_clients_ids (str): The client ids belonging to the first task.
        task_2_clients_ids (str): The client ids belonging to the second task.
        tasksplits ([x,x]]): How the data shall be split among task. For instance [0.5, 0.5] results in an equal split.

    Returns:
        DataManager: Returns a data manager for a specific dataset and its two task types.
    A data manager has train, val and test loaders for each client. The loaders are different for each client.
    There exists no overlap between train and validation data among clients. Test data is the same among clients belonging to the same task.
    """

    print("Getting Data Manager, This might take a few minutes")
    if dataset == "cifar10":
        data_manager = DMCifar(
            seed=1,
            an_num_clients=len(task_1_clients_ids),
            ob_num_clients=len(task_2_clients_ids),
            dataset_fraction=dataset_fraction,
        )  # datasetsplit should not sum up to more than 1!
    elif dataset == "celeba":
        data_manager = DMCelebA(
            seed=1,
            ml_num_clients=len(task_1_clients_ids),
            fl_num_clients=len(task_2_clients_ids),
            dataset_fraction=dataset_fraction,
        )  # datasetsplit should not sum up to more than 1!
    else:
        raise "Dataset not properly defined!"
    return data_manager
