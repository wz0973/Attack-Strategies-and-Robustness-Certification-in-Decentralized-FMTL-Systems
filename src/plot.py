import os
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Dict


def load_experiment_results(folder_path: str) -> List[Dict]:
    """
    Load all experiment checkpoints (.pth files) from a given folder.

    Args:
        folder_path (str): Path to the folder containing checkpoint files.

    Returns:
        List[Dict]: A list of checkpoint dictionaries loaded with torch.load().
    """
    results = []
    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".pth"):
            ckpt = torch.load(os.path.join(folder_path, file), map_location="cpu")
            results.append(ckpt)
    return results


def plot_metric_curves(results: List[Dict], task: str, output_folder: str):
    """
    Plot metric curves (loss, precision, recall, f1) for both training and validation phases.

    Args:
        results (List[Dict]): A list of experiment result dictionaries.
        task (str): Task identifier (e.g., "t1" or "t2").
        output_folder (str): Directory where the plots will be saved.
    """
    metrics = ["loss", "precision", "recall", "f1"]
    phases = ["train", "val"]

    fig, axs = plt.subplots(2, 4, figsize=(24, 10))
    axs = axs.flatten()

    for idx, (phase, metric) in enumerate([(p, m) for p in phases for m in metrics]):
        ax = axs[idx]
        for res in results:
            x = []
            y = []
            for m in res["metrics"]:
                if phase not in m or metric not in m[phase]:
                    print(f"⚠️ Warning: {res['c_id']} epoch {m['epoch']} lack {phase}.{metric}")
                    continue
                value = m[phase][metric]
                if isinstance(value, torch.Tensor):
                    value = value.detach().item()
                x.append(m["epoch"])
                y.append(value)
            if len(x) > 0:
                ax.plot(x, y, label=res["c_id"])
        ax.set_title(f"{phase.capitalize()} {metric.capitalize()} per Epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.capitalize())
        ax.grid(True)
        if task == "t1":
            ax.set_ylim(0, 1)
        ax.legend()

    plt.tight_layout()
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, f"Plot_{task}_all_metrics.png"))
    plt.close()


def plot_metric_table(results: List[Dict], task: str, output_folder: str):
    """
    Create a summary table of test metrics for each client.

    Args:
        results (List[Dict]): Experiment results list.
        task (str): Task identifier (e.g., "t1" or "t2").
        output_folder (str): Directory to save the table.

    Returns:
        pd.DataFrame: DataFrame with metrics per client.
    """
    metric_names = ["loss", "precision", "recall", "f1"]
    rows = []
    for res in results:
        row = [res["c_id"]]
        for m in metric_names:
            val = res["test_metrics"].get(m, None)
            if isinstance(val, torch.Tensor):
                val = val.detach().item()
            row.append(round(val, 3) if val is not None else "-")
        rows.append(row)

    df = pd.DataFrame(rows, columns=["Client"] + [m.capitalize() for m in metric_names])
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(df)))
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, f"Table_{task}.png"))
    plt.close()
    return df


def plot_cross_client_table(clean_df, poison_df, output_folder):
    """
    Generate a comparison table of Clean vs Poisoned results across all clients.

    Args:
        clean_df (pd.DataFrame): Metrics from clean experiment.
        poison_df (pd.DataFrame): Metrics from poisoned experiment.
        output_folder (str): Directory to save the table.
    """
    rows = []
    metric_names = ["Loss", "Precision", "Recall", "F1"]
    for i in range(len(clean_df)):
        row = [clean_df.iloc[i, 0]]
        for m in metric_names:
            row.append(clean_df.iloc[i][m])
            row.append(poison_df.iloc[i][m])
        rows.append(row)

    columns = ["Client"]
    for m in metric_names:
        columns.extend([f"{m}\n(Clean)", f"{m}\n(Poisoned)"])
    df = pd.DataFrame(rows, columns=columns)

    fig, ax = plt.subplots(figsize=(len(columns)*1.3, len(rows)*0.6))
    plt.title("Client Comparison", fontsize=14, pad=5)
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_height(0.2)
            cell.set_fontsize(11)
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, "Cross_Client_Table.png"))
    plt.close()


def plot_cross_client_bars(clean_df, poison_df, output_folder):
    """
    Plot bar charts comparing Clean vs Poisoned performance for each metric across clients.

    Args:
        clean_df (pd.DataFrame): Metrics from clean experiment.
        poison_df (pd.DataFrame): Metrics from poisoned experiment.
        output_folder (str): Directory to save plots.
    """
    metric_names = ["Loss", "Precision", "Recall", "F1"]
    clients = clean_df["Client"].tolist()
    x = np.arange(len(clients))
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Client Comparison: Clean vs Poisoned", fontsize=16, y=1.02)
    axs = axs.ravel()

    for i, metric in enumerate(metric_names):
        clean = [float(v) if str(v) != "-" else np.nan for v in clean_df[metric].tolist()]
        poison = [float(v) if str(v) != "-" else np.nan for v in poison_df[metric].tolist()]
        ax = axs[i]
        ax.bar(x - 0.2, clean, width=0.4, label="Clean")
        ax.bar(x + 0.2, poison, width=0.4, label="Poisoned")
        ax.set_xticks(x)
        ax.set_xticklabels(clients, rotation=45)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric}: Clean vs Poisoned")
        ax.legend(loc='lower right')
        ax.grid(True)

    plt.tight_layout()
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, "Bar_All_Metrics.png"), bbox_inches='tight')
    plt.close()


def auto_process_dataset(dataset_folder):
    """
    Process results for a dataset folder:
      - Load clean experiment results.
      - Load poisoned experiment results.
      - Generate metric tables, curves, comparison tables, and bar charts.

    Args:
        dataset_folder (str): Path to dataset results (e.g., results/cifar-10).
    """
    dataset_name = os.path.basename(dataset_folder)
    all_subdirs = [d for d in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, d))]
    clean_name = [d for d in all_subdirs if "POISON" not in d][0]
    poison_names = [d for d in all_subdirs if "POISON" in d]

    clean_t1 = load_experiment_results(os.path.join(dataset_folder, clean_name, "t1"))
    clean_t2 = load_experiment_results(os.path.join(dataset_folder, clean_name, "t2"))
    df_clean_t1 = plot_metric_table(clean_t1, "t1", os.path.join(dataset_folder, clean_name))
    df_clean_t2 = plot_metric_table(clean_t2, "t2", os.path.join(dataset_folder, clean_name))
    plot_metric_curves(clean_t1, "t1", os.path.join(dataset_folder, clean_name))
    plot_metric_curves(clean_t2, "t2", os.path.join(dataset_folder, clean_name))

    for poison in poison_names:
        poison_t1 = load_experiment_results(os.path.join(dataset_folder, poison, "t1"))
        poison_t2 = load_experiment_results(os.path.join(dataset_folder, poison, "t2"))
        df_poison_t1 = plot_metric_table(poison_t1, "t1", os.path.join(dataset_folder, poison))
        df_poison_t2 = plot_metric_table(poison_t2, "t2", os.path.join(dataset_folder, poison))
        plot_metric_curves(poison_t1, "t1", os.path.join(dataset_folder, poison))
        plot_metric_curves(poison_t2, "t2", os.path.join(dataset_folder, poison))

        df_clean_all = pd.concat([df_clean_t1, df_clean_t2], ignore_index=True)
        df_poison_all = pd.concat([df_poison_t1, df_poison_t2], ignore_index=True)
        outdir = os.path.join(dataset_folder, poison)
        plot_cross_client_table(df_clean_all, df_poison_all, outdir)
        plot_cross_client_bars(df_clean_all, df_poison_all, outdir)


def main():
    base_root = "results"
    datasets = ["cifar-10", "celeba"]
    for dataset in datasets:
        print(f"📊 Processing dataset: {dataset}")
        auto_process_dataset(os.path.join(base_root, dataset))


if __name__ == "__main__":
    main()
