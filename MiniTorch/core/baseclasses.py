import jax.numpy as jnp
import abc
import dataclasses
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import jax.random as jrandom
@dataclasses.dataclass
class ComputationNode(abc.ABC):
    '''
    Abstract base class for computation nodes in a neural network.

    Attributes:
    input : Input data to the node.
    output : Output data from the node.
    seed : Random seed for reproducibility.
    requires_grad : Boolean indicating if the node requires gradient computation.
    _grad_norm : Dictionary to store gradient norms.
    _accum_params : Dictionary to store accumulated parameters.
    _accum_param_var_mean : Dictionary to store variance and mean of accumulated parameters.
    grad_cache : Dictionary to cache gradients.
    accumulate_grad_norm : Boolean indicating if gradient norms should be accumulated.
    accumulate_parameters : Boolean indicating if parameters should be accumulated.
    '''
    
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
        '''
        Returns the variance and mean of the output.

        Returns:
        Tuple containing variance and mean of the output.
        '''
        try:
            return self.output.var(), self.output.mean()
        except Exception as e:
            return "Run the example through the layer"
    def in_var_mean(self):
        '''
        Returns the variance and mean of the input.

        Returns:
        Tuple containing variance and mean of the input.
        '''
        try:
            return self.input.var(), self.input.mean()
        except Exception as e:
            return "Run the example through the layer"
    def set_seed(self,seed):
        '''
        Sets the random seed for the node.

        Parameters:
        seed : Random seed value.
        '''
        self.seed = seed
    def _accumulate_grad_norm(self, grad_key):
        '''
        Accumulates the gradient norm for a given key.

        Parameters:
        grad_key : Key for the gradient to accumulate.
        '''
        if self.accumulate_grad_norm:
            if grad_key not in self._grad_norm:
                self._grad_norm[grad_key] = []
            self._grad_norm[grad_key].append(jnp.linalg.norm(self.grad_cache[grad_key].flatten()))

    def _accumulate_parameters(self, param_key, var_mean_func):
        '''
        Accumulates parameters using a variance and mean function.

        Parameters:
        param_key : Key for the parameter to accumulate.
        var_mean_func : Function to compute variance and mean.
        '''
        if self.accumulate_parameters:
            if param_key not in self._accum_param_var_mean:
                self._accum_param_var_mean[param_key] = {'var': [], 'mean': []}
            var, mean = var_mean_func()
            self._accum_param_var_mean[param_key]['var'].append(var)
            self._accum_param_var_mean[param_key]['mean'].append(mean)
    def plot_out(self):
        '''
        Plots a heatmap of the output for the first sample in the batch.
        '''
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
    def plot_kernel(self, kernel_index: int, cmap="coolwarm"):
        """
        Plots a specific kernel from the Conv2D layer as a heatmap.
        
        Args:
            kernel_index (int): Index of the kernel to plot.
            cmap (str): Colormap for the heatmap.
        """
        if not hasattr(self, 'parameters') or 'W' not in self.parameters:
            raise ValueError("No kernel weights found in the layer.")
        
        kernels = self.parameters['W']
        if kernel_index < 0 or kernel_index >= kernels.shape[0]:
            raise ValueError(f"Kernel index {kernel_index} is out of bounds. Expected index between 0 and {kernels.shape[0] - 1}.")
        
        kernel = kernels[kernel_index]
        plt.figure(figsize=(6, 6))
        sns.heatmap(kernel.squeeze(), cmap=cmap, annot=True, fmt=".2f")
        plt.title(f"Kernel {kernel_index} Heatmap")
        plt.show()
    def grad_check(self, input_data=None, output_grad=None, epsilon=1e-7, threshold=1e-5):
        """
        Performs gradient checking on the current layer by comparing analytical gradients 
        with numerical gradients computed via finite differences.
        
        Parameters:
        - input_data: Optional input data to use for the check. If None, uses the stored input.
        - output_grad: Optional output gradient to use for the check. If None, uses random gradients.
        - epsilon: Small perturbation value for finite difference calculation.
        - threshold: Threshold for relative error to consider gradient check passed.
        
        Returns:
        - dict: Dictionary containing gradient check results for each parameter and input.
        """
        
        if not hasattr(self, 'parameters') and not self.requires_grad:
            return {"status": "No parameters to check or layer doesn't require gradients"}
        
        # Use stored input if not provided
        if input_data is None:
            if self.input is None:
                return {"status": "No input data available. Run forward pass first or provide input_data."}
            input_data = self.input
        
        # Generate random output gradients if not provided
        if output_grad is None:
            key = jrandom.PRNGKey(0)
            # Forward pass to get output shape if needed
            output = self.forward(input_data)
            output_grad = jrandom.normal(key, output.shape)
        
        # Storage for results
        results = {}
        
        # Run forward and backward passes to get analytical gradients
        output = self.forward(input_data)
        self.backward(output_grad)
        
        # Check gradients for each parameter
        if hasattr(self, 'parameters'):
            for param_name, param_value in self.parameters.items():
                analytical_grad = self.grad_cache.get(f'dL_d{param_name}')
                if analytical_grad is None:
                    results[param_name] = {"status": f"No gradient found for {param_name}"}
                    continue
                    
                # Initialize numerical gradients array
                numerical_grad = jnp.zeros_like(param_value)
                
                # Compute numerical gradient for each element using finite differences
                it = jnp.nditer(param_value, flags=['multi_index'])
                while not it.finished:
                    idx = it.multi_index
                    
                    # Save original value
                    orig_val = param_value[idx].item()
                    
                    # Add epsilon
                    param_value_plus = param_value.at[idx].set(orig_val + epsilon)
                    self.parameters[param_name] = param_value_plus
                    output_plus = self.forward(input_data)
                    loss_plus = jnp.sum(output_plus * output_grad)
                    
                    # Subtract epsilon
                    param_value_minus = param_value.at[idx].set(orig_val - epsilon)
                    self.parameters[param_name] = param_value_minus
                    output_minus = self.forward(input_data)
                    loss_minus = jnp.sum(output_minus * output_grad)
                    
                    # Compute numerical gradient
                    numerical_grad = numerical_grad.at[idx].set((loss_plus - loss_minus) / (2 * epsilon))
                    
                    # Restore original value
                    self.parameters[param_name] = param_value.at[idx].set(orig_val)
                    
                    it.iternext()
                
                # Compute relative error
                abs_diff = jnp.abs(analytical_grad - numerical_grad)
                abs_norm = jnp.maximum(jnp.abs(analytical_grad), jnp.abs(numerical_grad))
                # Add small constant to avoid division by zero
                rel_error = jnp.mean(abs_diff / (abs_norm + 1e-10))
                
                results[param_name] = {
                    "analytical_grad": analytical_grad,
                    "numerical_grad": numerical_grad,
                    "relative_error": rel_error.item(),
                    "passed": rel_error < threshold,
                    "max_abs_diff": jnp.max(abs_diff).item()
                }
        
        # Also check input gradients if required
        if self.requires_grad:
            analytical_grad_input = self.grad_cache.get('dL_dinput')
            if analytical_grad_input is not None:
                numerical_grad_input = jnp.zeros_like(input_data)
                
                # Compute numerical gradient for input
                it = jnp.nditer(input_data, flags=['multi_index'])
                while not it.finished:
                    idx = it.multi_index
                    
                    # Save original value
                    orig_val = input_data[idx].item()
                    
                    # Add epsilon
                    input_data_plus = input_data.at[idx].set(orig_val + epsilon)
                    output_plus = self.forward(input_data_plus)
                    loss_plus = jnp.sum(output_plus * output_grad)
                    
                    # Subtract epsilon
                    input_data_minus = input_data.at[idx].set(orig_val - epsilon)
                    output_minus = self.forward(input_data_minus)
                    loss_minus = jnp.sum(output_minus * output_grad)
                    
                    # Compute numerical gradient
                    numerical_grad_input = numerical_grad_input.at[idx].set((loss_plus - loss_minus) / (2 * epsilon))
                    
                    # Restore original value
                    input_data = input_data.at[idx].set(orig_val)
                    
                    it.iternext()
                
                # Compute relative error
                abs_diff = jnp.abs(analytical_grad_input - numerical_grad_input)
                abs_norm = jnp.maximum(jnp.abs(analytical_grad_input), jnp.abs(numerical_grad_input))
                rel_error = jnp.mean(abs_diff / (abs_norm + 1e-10))
                
                results["input"] = {
                    "analytical_grad": analytical_grad_input,
                    "numerical_grad": numerical_grad_input,
                    "relative_error": rel_error.item(),
                    "passed": rel_error < threshold,
                    "max_abs_diff": jnp.max(abs_diff).item()
                }
        
        # Re-run forward and backward to restore original gradients
        self.forward(input_data)
        self.backward(output_grad)
        
        # Determine overall status
        all_passed = all(result.get("passed", False) for result in results.values() if "passed" in result)
        results["overall_status"] = "PASSED" if all_passed else "FAILED"
        
        return results
    
    def plot_grad_check_results(self, results):
        """
        Visualizes the results of gradient checking.
        
        Parameters:
        - results: Dictionary returned by grad_check method
        """
        
        if "status" in results:
            print(results["status"])
            return
        
        if "overall_status" not in results:
            print("No valid gradient check results to visualize")
            return
        
        # Print overall status
        status_color = "green" if results["overall_status"] == "PASSED" else "red"
        print(f"Overall Status: \033[1;{92 if status_color == 'green' else 91}m{results['overall_status']}\033[0m")
        
        # Count how many plots we need (parameters + input if checked)
        plot_items = [(name, data) for name, data in results.items() 
                    if name != "overall_status" and isinstance(data, dict) and "analytical_grad" in data]
        
        if not plot_items:
            print("No gradients to visualize")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(len(plot_items), 3, figsize=(18, 5 * len(plot_items)))
        
        # Handle case with only one parameter
        if len(plot_items) == 1:
            axes = axes.reshape(1, -1)
        
        for i, (name, result) in enumerate(plot_items):
            if "analytical_grad" not in result:
                continue
                
            analytical = np.array(result["analytical_grad"])
            numerical = np.array(result["numerical_grad"])
            
            # Plot analytical gradient
            if analytical.ndim <= 2:
                sns.heatmap(analytical, ax=axes[i, 0], cmap="coolwarm", annot=False)
            else:
                axes[i, 0].imshow(analytical[0], cmap="coolwarm")
            axes[i, 0].set_title(f"{name} - Analytical Gradient")
            
            # Plot numerical gradient
            if numerical.ndim <= 2:
                sns.heatmap(numerical, ax=axes[i, 1], cmap="coolwarm", annot=False)
            else:
                axes[i, 1].imshow(numerical[0], cmap="coolwarm")
            axes[i, 1].set_title(f"{name} - Numerical Gradient")
            
            # Plot absolute difference
            abs_diff = np.abs(analytical - numerical)
            if abs_diff.ndim <= 2:
                im = sns.heatmap(abs_diff, ax=axes[i, 2], cmap="Reds", annot=False)
            else:
                im = axes[i, 2].imshow(abs_diff[0], cmap="Reds")
            axes[i, 2].set_title(f"{name} - Absolute Difference")
            fig.colorbar(im, ax=axes[i, 2])
            
            # Add text with error information
            rel_error_color = "green" if result["passed"] else "red"
            axes[i, 2].text(0.5, -0.15, 
                            f"Relative Error: {result['relative_error']:.6f} ({'PASSED' if result['passed'] else 'FAILED'})", 
                            transform=axes[i, 2].transAxes,
                            ha='center', va='center',
                            color=rel_error_color, fontsize=12, fontweight='bold')
        
        plt.tight_layout(h_pad=2.0)
        plt.show()
    def get_svd(self):
        '''No idea where to put this function, for now here'''
        if self.__class__.__name__ == 'Linear':
            U, S, Vt = jnp.linalg.svd(self.W)
            print(f'Singular values of Linear_layer Weights -> {S}')
        elif self.__class__.__name__ == 'Conv2D':
            W = self.parameters['W']
            W = jnp.reshape(W,(W.shape[0],-1))
            U,S,vt = jnp.linalg.svd(W)
            print(f'Singular values of Conv2D_layer Kernels -> {S}')

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


