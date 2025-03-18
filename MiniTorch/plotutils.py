import matplotlib.pyplot as plt
import seaborn as sns

def show_conv_out(layer, sample, index):    
    kernels = layer.output[sample]
    kernel = kernels[index]
    plt.figure(figsize=(6, 6))
    sns.heatmap(kernel, cmap='coolwarm', fmt=".2f")
    plt.title(f"Kernel {index} output for sample {sample}")
    plt.show()