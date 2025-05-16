import torch
from schemas import EvaluateInput, EvaluateOutput
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from torch.nn.functional import conv2d
import os
import numpy as np
import pandas as pd
from pathlib import Path

#1. Defines evaluation task.

def off_mean(A):
    """
    Computes the mean of all elements off the primary diagonal of a square tensor A.
    """
    assert A.shape[0] == A.shape[1], "The tensor must be square"
    
    # Create a mask for the diagonal
    mask = ~torch.eye(A.shape[0], dtype=torch.bool, device=A.device)
    
    # Extract off-diagonal elements and compute mean
    return A[mask].mean()


def gaussian_kernel(patch_l2_norm, sigma=1.0):
    """Computes the Gaussian (RBF) kernel directly on patch-wise distances while preserving shape."""
    return torch.exp(-patch_l2_norm / (2 * sigma ** 2))  # Keeps shape

class PatchMMD(torch.nn.Module):
    """
    Computes patch-based pairwise loss using flexible kernels.
    Returns three pairwise distance matrices: D_xx, D_xy, D_yy.
    """
    def __init__(self, patch_size=3, channels=1, sigma=0.5, kernel_function=None):
        super(PatchMMD, self).__init__()
        self.patch_summation_kernel = torch.ones((1, channels, patch_size, patch_size), dtype=torch.float32)
        self.sigma = None if sigma is None else sigma * patch_size**2
        self.kernel_function = kernel_function if kernel_function is not None else gaussian_kernel
        self.name = f'SimplePatchLoss(p={patch_size}, s={sigma}, kernel={self.kernel_function.__name__})'

    def pairwise_patch_distance(self, A, B):
        """
        Computes pairwise patch-wise distances between each sample in A and each sample in B.
        Returns a tensor of shape (N_A, N_B, H_new, W_new) where each entry is the patch-wise distance.
        """
        N_A, C, H, W = A.shape
        N_B, _, _, _ = B.shape

        # Expand tensors to compute pairwise squared differences
        A_exp = A.view(N_A, 1, C, H, W)  # Shape: (N_A, 1, C, H, W)
        B_exp = B.view(1, N_B, C, H, W)  # Shape: (1, N_B, C, H, W)
        
        pixel_diff = (A_exp - B_exp) ** 2  # Pairwise squared differences: (N_A, N_B, C, H, W)

        # Apply patch summation using conv2d (batch-wise)
        patch_distances = conv2d(pixel_diff.flatten(0, 1), self.patch_summation_kernel.to(A.device))

        # Dynamically get the new height & width after conv2d
        H_new, W_new = patch_distances.shape[-2], patch_distances.shape[-1]

        return patch_distances.view(N_A, N_B, H_new, W_new)  # Correctly reshaped output

    def forward(self, x, y):
        # Compute all pairwise patch-wise distances
        D_xx = self.pairwise_patch_distance(x, x)  # (N_x, N_x, H_new, W_new)
        D_xy = self.pairwise_patch_distance(x, y)  # (N_x, N_y, H_new, W_new)
        D_yy = self.pairwise_patch_distance(y, y)  # (N_y, N_y, H_new, W_new)

    

        # Apply the kernel function
        K_xx = self.kernel_function(D_xx, self.sigma)  # (N_x, N_x, H_new, W_new)
        K_xy = self.kernel_function(D_xy, self.sigma)  # (N_x, N_y, H_new, W_new)
        K_yy = self.kernel_function(D_yy, self.sigma)  # (N_y, N_y, H_new, W_new)


        # Compute Maximum Mean Discrepancy (MMD) loss matrix
        loss = off_mean(K_xx) - 2 * torch.mean(K_xy) + off_mean(K_yy)  # (N_x, N_y, H_new, W_new)
        
        return loss  # Returning full pairwise loss matrix

def retrieve_sigma(samples, patch_size = 3):

    #b. Calculates sigma.
    metric_function = PatchMMD(patch_size= patch_size)
    pairwise_distances = metric_function.pairwise_patch_distance(samples, samples)
    sigma = pairwise_distances.flatten().median()

    print("Mean: {} - Median: {}".format(pairwise_distances.mean(), pairwise_distances.median()))

    return sigma



def single_permutation_test(tensor_a, tensor_b, distance_fn, num_permutations=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move tensors to GPU
    tensor_a = tensor_a.to(device)
    tensor_b = tensor_b.to(device)
    
    observed_stats = []
    null_stats = []
    
    for _ in tqdm(range(num_permutations), desc='Running permutations...'):

        # Shuffle tensor_a and tensor_b before computing the observed statistic
        permuted_a = tensor_a[torch.randperm(tensor_a.size(0))]
        permuted_b = tensor_b[torch.randperm(tensor_b.size(0))]
        
        # Observed statistic: compute distance after shuffling
        with torch.no_grad():
            obs_stat = distance_fn(permuted_a, permuted_b).mean()
        observed_stats.append(obs_stat)
        
        # Now create the null statistic by shuffling the combined bootstrapped samples.
        combined = torch.cat([permuted_a, permuted_b], dim=0)
        permuted_indices = torch.randperm(combined.size(0), device=device)
        permuted = combined[permuted_indices]
        
        # Split the permuted tensor into two groups of the same size as boot_a_flat and boot_b_flat
        mid_point = tensor_a.size(0)
        further_permuted_a = permuted[:mid_point]
        further_permuted_b = permuted[mid_point:]
        
        with torch.no_grad():
            null_stat = distance_fn(further_permuted_a, further_permuted_b).mean()
        null_stats.append(null_stat)
    
    observed_array = torch.stack(observed_stats)
    null_array = torch.stack(null_stats)
 
    r_factor = 1e-14
    p_value = (null_array.abs() >= observed_array.abs() - r_factor).float().mean().view(1, ).to('cpu').detach().numpy()

    return null_array.to('cpu').detach().numpy(), observed_array.to('cpu').detach().numpy(), p_value


def plot_permutation(null_dist, alt_dist, title="Permutation Test Distributions", save_path=os.getcwd(), bg_color='white'):
   
    # Plotting
    plt.figure(figsize=(10, 6))
    ax = plt.gca()  # Get current axis

    # Set background color inside the plot
    ax.set_facecolor('white')  # Light salmon pink background '#FAEDED'

    # Plot null distribution
    sns.kdeplot(null_dist, color='blue', shade=True, alpha=0.6, label='Null')
    # Plot alternative distribution
    sns.kdeplot(alt_dist, color='red', shade=True, alpha=0.6, label='Alternative')

    # Enhancing the graph
    plt.title('Permutation Test Distributions', fontsize=16)
    plt.xlabel('Distance', fontsize=14)
    plt.ylabel('Density', fontsize=14)

    # Removing grid lines
    plt.grid(False)

    # Legend with customized font size
    plt.legend(title='Hypothesis', title_fontsize='13', fontsize='12')

    # Save the figure with high resolution
    plt.savefig('permutation_test_distributions.png', dpi=300)

    # Save the plot if a save path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)

def plot_permutation_v2(null_dist, alt_dist, title="Permutation Test Distributions", bg_color='white'):
    """
    Plots the null and alternative distribution for a permutation test and optionally saves the plot.

    Parameters:
    - null_dist: The null distribution (array or list).
    - alt_dist: The alternative distribution (array or list).
    - title: The title of the plot (default is "Permutation Test Distributions").
    - save_path: The path to save the plot (default is the current working directory).
    - bg_color: The background color for the plot (default is 'white').

    Returns:
    - fig: The figure object containing the plot.
    """
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set background color inside the plot
    ax.set_facecolor(bg_color)

    # Plot null distribution
    sns.kdeplot(null_dist, color='blue', shade=True, alpha=0.6, label='Null', ax=ax)
    
    # Plot alternative distribution
    sns.kdeplot(alt_dist, color='red', shade=True, alpha=0.6, label='Alternative', ax=ax)

    # Enhancing the graph
    plt.title(title, fontsize=16)
    plt.xlabel('Distance', fontsize=14)
    plt.ylabel('Density', fontsize=14)

    # Removing grid lines
    plt.grid(False)

    # Legend with customized font size
    plt.legend(title='Hypothesis', title_fontsize='13', fontsize='12')

    # Return the figure object
    return fig


def evaluate_task(input: EvaluateInput) -> EvaluateOutput:
    """
    1. Retrieves paths for the test dataset and synthetic data.
    2. Collects n samples from the test loader.
    3. Applies bootstrapp to balance the number of samples.
    4. Performs permutation test. 
    5. Stores results: stats.csv, plot.png and raw results in csv.
    """

    #1. Retrieves paths. 
    _ , test_path = input.split_train_test
    test_loader = torch.load(str(test_path), weights_only = False)
    gen_samples = torch.load(str(input.gen_images_path))


    #2. Collects n samples from the test loader.
    test_samples = []
    for i in range(input.n_test_data):

        sample = next(iter(test_loader))[0]

        test_samples.append(next(iter(test_loader))[0])
    test_samples = torch.vstack(test_samples)

    #3. Bootstrapps data if necessary. 
    if input.bootstrap_data:
        #a. Test data (sometimes loaders possesses less than specified in n_test_data)
        if test_samples.shape[0] <= input.n_test_data:
            test_samples = test_samples[torch.randint(high = test_samples.shape[0], size = (input.n_test_data, ))]
        else:
            test_samples = test_samples[:input.n_test_data]

        #b. Applies bootstrap on generated data. 
        if gen_samples.shape[0] <= input.n_test_data:
            gen_samples = gen_samples[torch.randint(high = gen_samples.shape[0], size = (input.n_test_data, ))]

    #4. Performs permutation test.
    #a. Retrieves sigma for distance function.
    sigma = retrieve_sigma(samples = test_samples, patch_size= input.kernel_patch_size)
    distance_fn = PatchMMD(patch_size= input.kernel_patch_size, sigma= sigma)

    #b. Instantiates permutation. 
    null_array, obs_array, p_value = single_permutation_test(test_samples, gen_samples, 
                                                             distance_fn= distance_fn, num_permutations= input.num_permutations)
    
    #5. Stores results.

    evaluate_name = input.evaluate_results_name
    eval_dir = Path("Evaluation Results/{}".format( evaluate_name))
    eval_dir.mkdir(parents= True, exist_ok= True)

    #a. Table statistics. 
    null_stats = pd.DataFrame(null_array).describe(percentiles= [0.25, 0.5, 0.75, 0.9])
    obs_stats = pd.DataFrame(obs_array).describe(percentiles= [0.25, 0.5, 0.75, 0.9])

    perm_stats = pd.concat([null_stats, obs_stats], axis =1)
    perm_stats.columns = ["Null", "Obs."]

    perm_stats_path = eval_dir / "perm_stats.csv"
    perm_stats.to_csv(str(perm_stats_path))

    #b. Permutation plot. 
    perm_plot = plot_permutation_v2(null_dist= null_array, alt_dist =obs_array)
    perm_plot_path = eval_dir / "perm_plot.png"
    perm_plot.savefig(str(perm_plot_path), dpi = 300)
    
    return EvaluateOutput( stats_csv= perm_stats_path, plot_png= perm_plot_path)
