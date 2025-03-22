import matplotlib.pyplot as plt
import seaborn as sns

def show_conv_out(layer, sample, index):
    '''
    Visualizes the output of a convolutional layer for a specific sample and kernel index.

    Parameters:
    layer : The convolutional layer whose output is to be visualized.
    sample : The index of the sample in the batch.
    index : The index of the kernel in the layer.
    '''
    kernels = layer.output[sample]
    kernel = kernels[index]
    plt.figure(figsize=(6, 6))
    sns.heatmap(kernel, cmap='coolwarm', fmt=".2f")
    plt.title(f"Kernel {index} output for sample {sample}")
    plt.show()