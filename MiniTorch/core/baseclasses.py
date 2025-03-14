import jax.numpy as jnp
import abc
import dataclasses
import matplotlib.pyplot as plt
import seaborn as sns

@dataclasses.dataclass
class ComputationNode(abc.ABC):
    
    input = None
    output = None
    seed = None
    requires_grad : bool= True
    _grad_norm : dict= dataclasses.field(default_factory=lambda : {})
    _accum_params : dict = dataclasses.field(default_factory=lambda : {})
    _accum_param_var_mean : dict = dataclasses.field(default_factory=lambda : {})
    grad_cache : dict = dataclasses.field(default_factory=lambda : {})
    accumulate_grad_norm : bool = False
    accumulate_parameters : bool = False
    def out_var_mean(self):
        try:
            return self.output.var(), self.output.mean()
        except Exception as e:
            return "Run the example through the layer"
    def in_var_mean(self):
        try:
            return self.input.var(), self.input.mean()
        except Exception as e:
            return "Run the example through the layer"
    def set_seed(self,seed):
        self.seed = seed
    def _accumulate_grad_norm(self, grad_key):
        if self.accumulate_grad_norm:
            if grad_key not in self._grad_norm:
                self._grad_norm[grad_key] = []
            self._grad_norm[grad_key].append(jnp.linalg.norm(self.grad_cache[grad_key].flatten()))

    def _accumulate_parameters(self, param_key, var_mean_func):
        if self.accumulate_parameters:
            if param_key not in self._accum_param_var_mean:
                self._accum_param_var_mean[param_key] = {'var': [], 'mean': []}
            var, mean = var_mean_func()
            self._accum_param_var_mean[param_key]['var'].append(var)
            self._accum_param_var_mean[param_key]['mean'].append(mean)
    def plot_out(self):
        """Plots a heatmap of the output for the first sample in the batch."""
        output = self.output[0] if self.output.ndim == 3 else self.output
        plt.figure(figsize=(12, 5))
        sns.heatmap(output, cmap="coolwarm", annot=True, fmt=".2f")
        plt.title("Linear Layer Output Heatmap")
        plt.show()
    def plot_grad_dist(self):
        '''
        Plots the Gradients distribution
        '''
        if not self.grad_cache:
            return "No Gradients available to plot"
        
        no_of_plots = len(self.grad_cache)
        fig, axes = plt.subplots(1, no_of_plots, figsize=(5 * no_of_plots, 4))

        if no_of_plots == 1:
            axes = [axes]
        for idx, (grad_name, grad_values) in enumerate(self.grad_cache.items()):
            axes[idx].hist(grad_values.flatten(), bins=50, alpha=0.7)
            axes[idx].set_title(f"Gradient Distribution: {self.__class__.__name__} - {grad_name}")
            axes[idx].set_xlabel("Gradient Value")
            axes[idx].set_ylabel("Frequency")
        
        plt.tight_layout()
        plt.show()
    def plot_grad_norm(self):
        '''
        Plots the Gradient Norm over training epochs for parameters
        '''
        if not self._grad_norm:
            return "No Gradients norms are not available for plotting, set accumulation to true while training"
        
        no_of_plots = len(self._grad_norm)
        fig, axes = plt.subplots(1, no_of_plots, figsize=(5 * no_of_plots, 4))

        if no_of_plots == 1:
            axes = [axes]
        for idx, (grad_name, grad_norms) in enumerate(self._grad_norm.items()):
            axes[idx].plot(range(len(grad_norms)),grad_norms, c = 'b')
            axes[idx].set_title(f"Gradient Norm: {self.__class__.__name__} - {grad_name}")
            axes[idx].set_xlabel("Epochs")
            axes[idx].set_ylabel("Gradient Norm")
        plt.tight_layout()
        plt.show()
    def plot_param_var_mean(self):
        '''
        Plots the accumulated variance and mean of the parameters over training epochs.
        '''
        if not self._accum_param_var_mean:
            return "No parameter variance/mean data available for plotting. Set accumulate_parameters=True during training."
        
        no_of_plots = len(self._accum_param_var_mean)
        fig, axes = plt.subplots(2, no_of_plots, figsize=(5 * no_of_plots, 8))

        if no_of_plots == 1:
            axes = axes[:, None]  # Ensure axes is 2D for consistent indexing
        
        for idx, (param_key, data) in enumerate(self._accum_param_var_mean.items()):
            # Plot variance
            axes[0, idx].plot(range(len(data['var'])), data['var'], c='r', label='Variance')
            axes[0, idx].set_title(f"Parameter Variance: {self.__class__.__name__} - {param_key}")
            axes[0, idx].set_xlabel("Epochs")
            axes[0, idx].set_ylabel("Variance")
            axes[0, idx].legend()
            
            # Plot mean
            axes[1, idx].plot(range(len(data['mean'])), data['mean'], c='g', label='Mean')
            axes[1, idx].set_title(f"Parameter Mean: {self.__class__.__name__} - {param_key}")
            axes[1, idx].set_xlabel("Epochs")
            axes[1, idx].set_ylabel("Mean")
            axes[1, idx].legend()
        
        plt.tight_layout()
        plt.show()
    def grad_check(self):
        if not hasattr(self, 'parameters'):
            raise ValueError
        for parameter in self.parameters:
            pass


        
    @abc.abstractmethod
    def forward(self, input):
        raise NotImplementedError
    @abc.abstractmethod
    def backward(self, output_grad):
        raise NotImplementedError
    
class Loss(abc.ABC):

    def __init__(self):
        self.input = None
        self.output = None
        self.grad_cache = {}
    @abc.abstractmethod
    def loss(self, pred, true):
        raise NotImplementedError
    @abc.abstractmethod
    def backward(self,output_grad):
        raise NotImplementedError
    
class Optimizer(abc.ABC):
    def __init__(self):
        self.params = None
        self.net = None
    @abc.abstractmethod
    def step(self):
        raise NotImplementedError


