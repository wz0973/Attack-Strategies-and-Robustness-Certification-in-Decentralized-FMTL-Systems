import copy
import os
from client_handling.client_manager import get_clients
from data_handling.data_manager import get_data_manager
from client_handling.seed import set_seed
import torch
import yaml
import argparse
import numpy as np
import random

def _print_experiment_information(
    general_config, tasks_config, task_1_clients_ids, task_2_clients_ids
):
    print("-----------------------------------")
    print(f'Running Experiment: {general_config["title"]}')
    print(f'- Dataset: {general_config["dataset"]}')
    print(f"- intra_communication: {general_config['intra_communication']}")
    print(f"- cross_communication: {general_config['cross_communication']}")
    print("-----------------------------------")
    print(f"Task: {tasks_config[0]['task']}, Epochs: {tasks_config[0]['epochs']}")
    print(f"Clients: {task_1_clients_ids}")
    print("-----------------------------------")
    print(f"Task: {tasks_config[1]['task']}, Epochs: {tasks_config[1]['epochs']}")
    print(f"Clients: {task_2_clients_ids}")
    return


def _get_client_ids(tasks_config):
    """
    Categorizes clients into two task groups based on task type.
    Returns:
    (list, list): task_1_clients_ids, task_2_clients_ids
    """
    print("Getting Client Ids")
    task_1_clients_ids = []
    task_2_clients_ids = []

    # Iterate over each task and categorize clients by task name
    for task in tasks_config:
        if (
            task["task"] == "Multi_Label"
            or task["task"] == "Animals"
            or task["task"] == "Binary_Classification"
        ):
            task_1_clients_ids.extend(task["clients"])
        elif (
            task["task"] == "Facial_Landmark"
            or task["task"] == "Objects"
            or task["task"] == "Segmentation"
        ):
            task_2_clients_ids.extend(task["clients"])
    return task_1_clients_ids, task_2_clients_ids

def _conduct_experiment(
    t1_clients,
    t2_clients,
    intra,
    cross,
    rounds,
    t1_epochs,
    t2_epochs,
    hca_alpha,
    results_folder,
):
    """
    Core training loop of the federation experiment.
    Handles local training, intra-task communication, and cross-task communication.
    """

    # Iterate over all communication rounds
    for round in range(0, rounds):
        print(f"🎯Local Round {round} started")
        # Perform last round if the last communication round has been reached
        if round == rounds - 1:
            print(
                "This is the last round - Model will only train locally and then conduct test"
            )
            _final_round_and_model_saving(
                t1_clients,
                t2_clients,
                t1_epochs,
                t2_epochs,
                round,
                results_folder,
            )
        # Perform standard training round if the last communication round has not been reached
        else:
            # Let each client of Task 1 train for the amount of epochs
            print("---Training on T1---")
            for client in t1_clients:
                print(f"-Client {client.c_id}")
                client.conduct_training(num_epochs=t1_epochs, current_round=round)

            # Let each client of Task 2 train for the amount of epochs
            print("---Training on T2---")
            for client in t2_clients:
                print(f"-Client {client.c_id}")
                client.conduct_training(num_epochs=t2_epochs, current_round=round)

            # If intra communication is FedPer perform the intra taskgroup communication round which averagaes the backbone of all clients of a task group
            # Here the communication (across network and between clients) is not modeled to make it run on a single device
            if intra == "FedPer":
                print("---Intra Task Aggregation of T1 with FedPer---")
                _intra_task_backbone_averaging(t1_clients)
                are_different = _are_state_dict_different(
                    t1_clients[0].curr_backbone, t1_clients[0].prev_backbone
                )
                if are_different == False:
                    raise "There has been an error during training. The Current and Previous Backbone are not different."

                print("---Intra Task Aggregation of T2 with FedPer---")
                _intra_task_backbone_averaging(t2_clients)
                are_different = _are_state_dict_different(
                    t2_clients[0].curr_backbone, t2_clients[0].prev_backbone
                )
                if are_different == False:
                    raise "There has been an error during training. The Current and Previous Backbone are not different."

            # If cross communication is FedPer average the backbone of both client task groups.
            # Since FedPer has been performed within the task group one can simply take the average of the task group leader,
            # which here is the first in the client list.
            # Here the communication (across network and between clients) is not modeled to make it run on a single device
            if cross == "FedPer":
                print("---Cross Task Aggregation of T1 and T2 with FedPer---")
                averaged_backbone = t1_clients[0].average_backbones(
                    t1_clients[0].curr_backbone_avg_with_neighbors,
                    [t2_clients[0].curr_backbone_avg_with_neighbors],
                )

                # Replace the backbone of all clients
                for client in t1_clients:
                    client.replace_backbone(averaged_backbone)
                for client in t2_clients:
                    client.replace_backbone(averaged_backbone)

            # If cross communicationis FedPer perform hca on the backbone of both client task groups.
            # Since FedPer has been performed within the task group one can simply take the average of the task group leader,
            # which here is the first in the client list.
            # Here the communication (across network and between clients) is not modeled to make it run on a single device
            if cross == "HCA":
                print("---Cross Task Aggregation of T1 and T2 with HCA---")

                curr_averaged_backbones_dicts = [
                    t1_clients[0].curr_backbone_avg_with_neighbors,
                    t2_clients[0].curr_backbone_avg_with_neighbors,
                ]
                prev_averaged_backbones_dicts = [
                    t1_clients[0].prev_backbone_avg_with_neighbors,
                    t2_clients[0].prev_backbone_avg_with_neighbors,
                ]

                # Perform HCA by using the client leader
                hca_backbones = t1_clients[0].conflict_averse(
                    curr_averaged_backbones_dicts,
                    prev_averaged_backbones_dicts,
                    hca_alpha,
                    title_of_experiment=t1_clients[0].title_of_experiment,
                )

                # Replace the backbone of all clients
                _replace_backbones_with_hca_backbones(t1_clients, hca_backbones[0])
                _replace_backbones_with_hca_backbones(t2_clients, hca_backbones[1])


def _final_round_and_model_saving(
    t1_clients, t2_clients, t1_epochs, t2_epochs, round, results_folder
):
    """
    Function that performs the last round of training and saves the models afterwards
    """
    print("---Training on T1---")
    for client in t1_clients:
        print(f"-Client {client.c_id}")
        client.conduct_training(num_epochs=t1_epochs, current_round=round)

    print("---Training on T2---")
    for client in t2_clients:
        print(f"-Client {client.c_id}")
        client.conduct_training(num_epochs=t2_epochs, current_round=round)

    print("---Testing on T1---")
    for client in t1_clients:
        print(f"-Client {client.c_id}")
        client.conduct_testing()

    print("---Testing on T2---")
    for client in t2_clients:
        print(f"-Client {client.c_id}")
        client.conduct_testing()

    print("---Saving all models---")
    for client in t1_clients:
        os.makedirs(results_folder, exist_ok=True)
        _filepath = os.path.join(results_folder, "t1")
        os.makedirs(_filepath, exist_ok=True)
        filepath = os.path.join(_filepath, f"model{client.c_id}.pth")
        torch.save(client.current_checkpoint, filepath)
    for client in t2_clients:
        os.makedirs(results_folder, exist_ok=True)
        _filepath = os.path.join(results_folder, "t2")
        os.makedirs(_filepath, exist_ok=True)
        filepath = os.path.join(_filepath, f"model{client.c_id}.pth")
        torch.save(client.current_checkpoint, filepath)


def _replace_backbones_with_hca_backbones(all_clients, hca_backbone):
    """
    Replaces each client's backbone with HCA-aggregated backbone.

    Args:
        all_clients ([clients]]): Clients for which the backbone shall be replaced
        hca_backbone (backbone): the new backbone after hca has been performed
    """
    for idx, client in enumerate(all_clients):
        client.replace_backbone(hca_backbone)


def _intra_task_backbone_averaging(clients):
    """
    Helper Function to perform the cross task averaging
    """
    for client in clients:
        client.prev_neighbour_backbones = []
        client.curr_neighbour_backbones = []
        # Inner loop to iterate over all other clients except the current one
        for other_client in clients:
            if other_client != client:  # Exclude the current client from processing
                client.prev_neighbour_backbones.append(
                    copy.deepcopy(other_client.prev_backbone)
                )
                client.curr_neighbour_backbones.append(
                    copy.deepcopy(other_client.curr_backbone)
                )
    for (
        client
    ) in (
        clients
    ):  # Iterating has to be done twice as otherwise the backbones of some clients will already be replaced witht their average
        avg_backbone = client.average_backbones(
            client.curr_backbone, client.curr_neighbour_backbones
        )
        client.prev_backbone_avg_with_neighbors = client.average_backbones(
            client.prev_backbone, client.prev_neighbour_backbones
        )
        client.curr_backbone_avg_with_neighbors = avg_backbone
        client.replace_backbone(avg_backbone)


def _are_state_dict_different(state_dict1, state_dict2):
    """
    Checks if two state_dicts differ (based on first key's tensor).
    """
    # Get the first key from each state_dict
    first_key1 = next(iter(state_dict1))
    first_key2 = next(iter(state_dict2))

    # Compare the tensors associated with the first key
    if torch.equal(state_dict1[first_key1], state_dict2[first_key2]):
        return False  # Return False if they are the same
    else:
        return True  # Return True if they are different
    
def run(exp_config, device):
    """
    Runs the federation by handling all the individual steps, indcluding: condig loading, client creation, data management, running the experiment
    """

    # Load the configuration yaml files
    config_path = os.path.join(os.path.dirname(__file__), "configs", exp_config)
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Create Sub-Configs
    general_config = config["general"]
    setup_config = config["setup"]
    tasks_config = setup_config["tasks"]

    # Set the seed of the experiment. 
    # This is done here instead of the start of the main function so that if an experiment is run independetly it still generates the same result.
    set_seed(general_config["seed"])

    # Create the results folder and duplicate the configuration file into the results folder
    results_folder = os.path.join(
        general_config["result_folder"], general_config["result_sub_folder"], general_config["title"]
    )
    os.makedirs(results_folder, exist_ok=True)
    filename = os.path.join(results_folder, "config.txt")
    with open(filename, "w") as text_file:
        yaml.dump(config, text_file)

    # Get all the client ids per task
    task_1_clients_ids, task_2_clients_ids = _get_client_ids(tasks_config)

    # Get the dataset specific datamanager
    data_manager = get_data_manager(
        general_config["dataset"],
        task_1_clients_ids,
        task_2_clients_ids,
        setup_config["dataset_fraction"],
    )

    # Get the clients for task 1 and 2
    t1_clients, t2_clients = get_clients(
        general_config["dataset"],
        general_config["title"],
        tasks_config[0]["task"],
        tasks_config[1]["task"],
        data_manager,
        task_1_clients_ids,
        task_2_clients_ids,
        device,
        general_config["backbone_layers"],
    )

    # Print the experiment Information
    _print_experiment_information(
        general_config, tasks_config, task_1_clients_ids, task_2_clients_ids
    )

    # Run the experiment
    _conduct_experiment(
        t1_clients,
        t2_clients,
        general_config["intra_communication"],
        general_config["cross_communication"],
        general_config["rounds"],
        tasks_config[0]["epochs"],
        tasks_config[1]["epochs"],
        setup_config["hca_alpha"],
        results_folder,
    )

def main():
    """This is the main function and entry point for running a federation."""

    # Set up
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")
    print("PyTorch version:", torch.__version__)
    print("CUDA version used by PyTorch:", torch.version.cuda)

    # This has to be set to enable determins with loss.backward. Otherwise an error gets thrown when torch.use_deterministic_algorithms(True) is set.
    # Read here for more information: https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Make the execution deterministic
    # The seed will be set for each config file independently at the start of the run function.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    # Entry point arguments
    parser = argparse.ArgumentParser(description="Entry Point to Train a Federation")
    parser.add_argument(
        "--configs_folder", required=True, help="Folder of Config Files to run"
    )
    parser.add_argument(
        "--configs", nargs="+", required=True, help="List of configuration files to run"
    )
    args = parser.parse_args()

    # Extract the paths for the configuration files
    config_paths = []

    for config in args.configs:
        c = os.path.join(args.configs_folder, config)
        config_paths.append(os.path.join(os.getcwd(), c))
        

    # Run the federation
    for config in config_paths:
        run(config, device)

if __name__ == "__main__":
    main()
