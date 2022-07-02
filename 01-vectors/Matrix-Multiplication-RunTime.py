"""
Run time performance
for loop vs NumPy vs Tensor
"""

# Import libraries
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns


# Compute matrix multiplication using for loops
def matrix_mult_loop(a_matrix, b_matrix):
    """
    Using for loops to perform matrix multiplication
    """
    matrix = np.zeros((len(a_matrix[0]), len(b_matrix[1])))
    for i in range(len(a_matrix)):
        for j in range(len(b_matrix[i])):
            matrix[i][j] = sum(a_matrix[i, :] * b_matrix[:, j])

    return np.round(matrix, 2)


# Using NumPy
def np_matrix_mult(a_matrix: np.array, b_matrix: np.array):
    """
    Using NumPy to perform matrix multiplication
    :type a_matrix: np.array
    :type b_matrix: np.array

    """
    return np.round(np.matmul(a_matrix, b_matrix))


# Using PyTorch Tensor
def tensor_matrix_mult(a_matrix, b_matrix):
    """
    Using PyTorch tensors to perform matrix multiplication
    """
    return torch.round(torch.matmul(a_matrix, b_matrix))


# Compute run time
def compute_time(func, a_matrix, b_matrix, sizes):
    """
    Function that tracks run time for each multiplication method
    """
    # Run time list
    run_time = []
    for s in sizes:
        # Start run
        start = time.time()

        # Perform matrix multiplication
        for _ in range(s):
            run = func(a_matrix, b_matrix)

        # End run
        end = time.time()
        run_time.append(round(end - start, 3))

    return run_time


# Plot Linear Regression graph
def plot_graph(sizes, loop_var, numpy_var, tensor_var):
    """
    Function that plots a graph to compare run time performance between loops, numpy and tensors.
    """
    plt.figure(figsize=(10, 6), dpi=600)
    plt.title('Run Time performance')
    plt.xlabel("Iterations")
    plt.ylabel("Run time (seconds)", rotation=90, labelpad=15)
    sns.lineplot(x=sizes, y=loop_var, color='green', label='Loop run time', marker='o')
    sns.lineplot(x=sizes, y=numpy_var, color='blue', label='Numpy run time', marker='v')
    sns.lineplot(x=sizes, y=tensor_var, color='red', label='Tensor run time', marker='X')
    plt.grid(color='blue', linestyle='--', linewidth=1, alpha=0.2)
    plt.legend(loc='upper left')
    # plt.savefig('Linear_Regression.jpg')
    plt.show()


# Function that prints the results
def print_results(loop_var, numpy_var, tensor_var):
    # Print results
    print(f"Loop run time:    {loop_var}")
    print(f"NumPy run time:   {numpy_var}")
    print(f"Tensor run time:  {tensor_var}")


# Main function
def main(sizes):
    """
    Function that performs run time computations
    return: loop_run, numpy_run, tensor_run
    """
    # Define matrices
    m = 10
    n = 10
    A = np.round(np.random.randn(m, n), 2)
    B = np.round(np.random.randn(n, m), 2)

    # Compute run time
    loop_var = compute_time(func=matrix_mult_loop, a_matrix=A, b_matrix=B, sizes=sizes)
    numpy_var = compute_time(func=np_matrix_mult, a_matrix=A, b_matrix=B, sizes=sizes)
    tensor_var = compute_time(func=tensor_matrix_mult, a_matrix=torch.tensor(A), b_matrix=torch.tensor(B), sizes=sizes)

    return loop_var, numpy_var, tensor_var


# # Call main function
# if __name__ == '__main__':
#     # Define iteration sizes
#     # iterations = [i for i in range(0, 51000, 5000)]
#     iterations = [1000]

#     # Compute run-time performance
#     loop_run, numpy_run, tensor_run = main(sizes=iterations)

#     # Print results
#     print_results(loop_var=loop_run, numpy_var=numpy_run, tensor_var=tensor_run)

#     # Plot graph
#     # plot_graph(sizes=iterations, loop_var=loop_run, numpy_var=numpy_run, tensor_var=tensor_run)
