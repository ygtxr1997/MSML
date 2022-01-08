import numpy as np

import torch

import matplotlib
import matplotlib.pyplot as plt


def plot_tensor(tensor_list: tuple,
                title_list: tuple,
                img_name: str,
                x_labels: tuple = (),
                y_labels: tuple = (),
                ):
    # 0. Get settings
    num_plot = len(tensor_list)
    assert num_plot <= 4
    assert len(tensor_list) == len(title_list)

    flag_x_labels = True if len(x_labels) == len(tensor_list) else False
    flag_y_labels = True if len(y_labels) == len(tensor_list) else False

    height, width = tensor_list[0].shape
    for tensor in tensor_list:
        assert len(tensor.size()) == 2
        assert height, width == tensor.shape

    # 1. Create window
    fig, ax = plt.subplots(figsize=(10 * num_plot, 10),
                           nrows=1, ncols=num_plot)
    ax = (ax, ) if num_plot == 1 else ax

    # 2. Plot each subplot
    for ind in range(num_plot):
        # Convert to numpy.array
        array = tensor_list[ind].detach().numpy()

        # heat map
        im = ax[ind].imshow(array, )

        # Edit labels and label them with the respective list entries
        if flag_x_labels:
            ax[ind].set_xticks(np.arange(len(x_labels)))
            ax[ind].set_xticklabels(x_labels)
            # Rotate the tick labels and set their alignment.
            plt.setp(ax[ind].get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor")
        if flag_y_labels:
            ax[ind].set_yticks(np.arange(len(y_labels)))
            ax[ind].set_yticklabels(y_labels)

        # Loop over data dimensions and create text annotations.
        for i in range(height):
            for j in range(width):
                text = ax[ind].text(j, i, '{:.2f}'.format(array[i, j]),
                               ha="center", va="center", color="k",
                                    fontsize=30)

        ax[ind].set_title(title_list[ind])

    fig.tight_layout()
    # plt.colorbar(im)
    plt.savefig(img_name)

if __name__ == '__main__':

    test_tensors = (torch.randn(6, 8),
                    torch.randn(6, 8),
                    torch.randn(6, 8),
                    )

    plot_tensor(test_tensors,
                ('test1', 'test2', 'test3'),
                'hello.jpg',
                )

