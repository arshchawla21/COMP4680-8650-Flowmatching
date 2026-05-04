import dataloader as dl
import torch
import matplotlib.pyplot as plt

def scatter_plot(data1, data2, title1="Dataset 1", title2="Dataset 2", suptitle=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot
    axes[0].scatter(data1[:, 0], data1[:, 1], alpha=0.5, s=10)
    axes[0].set_title(title1)
    axes[0].set_xlabel("Dimension 1")
    axes[0].set_ylabel("Dimension 2")
    axes[0].grid(True)

    # Right plot
    axes[1].scatter(data2[:, 0], data2[:, 1], alpha=0.5, s=10)
    axes[1].set_title(title2)
    axes[1].set_xlabel("Dimension 1")
    axes[1].set_ylabel("Dimension 2")
    axes[1].grid(True)

    if suptitle:
        fig.suptitle(suptitle)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    sz = [2, 32]

    for dataset in dl.AVAILABLE_DATASETS:
        dataloader_low = dl.get_dataloader(name=dataset, dim=sz[0])
        image_low = next(iter(dataloader_low))

        dataloader_high = dl.get_dataloader(name=dataset, dim=sz[1])
        image_high = next(iter(dataloader_high))

        dataloader = dl.ToyDiffusionDataset(dataset, dim=sz[1])
        image_high_2d = dataloader.to_2d(image_high)

        scatter_plot(image_low, image_high_2d, f"Visualise {dataset} with D={sz[0]}", f"Visualise {dataset} with D={sz[1]}")