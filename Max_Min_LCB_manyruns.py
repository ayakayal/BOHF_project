import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel, RBF, Hyperparameter, Matern
from sklearn.metrics.pairwise import pairwise_kernels
from itertools import product
from mpl_toolkits.mplot3d import Axes3D
import datetime
import os
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
import wandb
import argparse

def scale_to_range(data, new_min, new_max):
    old_min, old_max = np.min(data), np.max(data)
    return new_min + (data - old_min) * (new_max - new_min) / (old_max - old_min)
############### added ackley function #################################################
def ackley_function_1d(x, a=20, b=0.2, c=2*np.pi):
    sum_sq_term = -a * np.exp(-b * np.sqrt(x**2))
    cos_term = -np.exp(np.cos(c * x))
    
    return a + np.exp(1) + sum_sq_term + cos_term
################################################# Generate ackley function ##########################
def generate_ackley(grid_size=100): 
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate grid values between -5 and 5
    values = np.linspace(-5, 5, grid_size)
    
    # Initialize the matrix f with zeros
    f = np.zeros((grid_size, grid_size))
    
    # Compute f using the Ackley function
    for i in range(grid_size):
        for j in range(grid_size):
            f[i, j] = ackley_function_1d(values[i])-ackley_function_1d(values[j])
    
    #f=5*f
    f = scale_to_range(f, -3, 3)
    #print('f',f)
    # Create a figure with two subplots
    # fig = plt.figure(figsize=(14, 7))
    # # Compute 1D Ackley function values
    # ackley_values = ackley_function_1d(values)
    # # Plot 1D Ackley function
    # ax1 = fig.add_subplot(121)
    # ax1.plot(values, ackley_values, color='b')
    # ax1.set_xlabel(r"$x$", fontsize=14)
    # ax1.set_ylabel(r"$f(x)$", fontsize=14)
   
    # ax1.grid(False)
    # ax1.legend()
    
    # # Plot 2D preference function
    # ax2 = fig.add_subplot(122, projection='3d')
    # state_mesh, action_mesh = np.meshgrid(values, values)
    # surf = ax2.plot_surface(state_mesh, action_mesh, f, cmap='viridis')
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # ax2.set_xlabel('x1', fontsize=14)
    # ax2.set_ylabel('x2', fontsize=14)
    # ax2.set_zlabel('f(x1, x2)', fontsize=14)
    # ax2.set_title('2D Preference Function', fontsize=16)
    # ax2.view_init(elev=30, azim=45)

    # # # Log f image to WandB
    # wandb.log({"f Image": wandb.Image(fig)})
    # # save_path = os.path.join(f"testing1/f_{timestamp}.png")
    # # plt.savefig(save_path)
    # plt.close()
    # Plot 1D Ackley function
    fig1 = plt.figure(figsize=(7, 7))
    ackley_values = ackley_function_1d(values)  # Compute 1D Ackley function values
    plt.plot(values, ackley_values, color='b')
    plt.xlabel(r"$x$", fontsize=14)
    plt.ylabel(r"$f(x)$", fontsize=14)
    plt.grid(False)
    plt.legend()
    # Log 1D Ackley function figure to WandB
    wandb.log({"1D Ackley Function": wandb.Image(fig1)})
    plt.close(fig1)  # Close the figure after logging

    # Plot 2D Preference function
    fig2 = plt.figure(figsize=(7, 7))
    action_mesh, state_mesh = np.meshgrid(values, values)
    ax2 = fig2.add_subplot(111, projection='3d')
    surf = ax2.plot_surface(state_mesh, action_mesh, f, cmap='viridis')
    fig2.colorbar(surf, shrink=0.5, aspect=5)
    ax2.set_xlabel(r"$x$", fontsize=20)
    ax2.set_ylabel(r"$x^{'}$", fontsize=20)
    ax2.set_zlabel(r"$h(x, x^{'})$", fontsize=20)
    ax2.grid(False)
   
    ax2.view_init(elev=30, azim=45)
    # Log 2D Preference function figure to WandB
    wandb.log({"2D Preference Function": wandb.Image(fig2)})
    plt.close(fig2)  # Close the figure after logging
    
    # Calculate sigmoid of f
    sigmoid_f = sigmoid(f)

    # Plotting sigmoid(f)
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(state_mesh, action_mesh, sigmoid_f, cmap='viridis')
    fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.set_xlabel(r"$x$", fontsize=20)
    ax.set_ylabel(r"$x^{'}$", fontsize=20)
    ax.set_zlabel(r"$\mu(h(x, x^{'}))$", fontsize=20)
    ax.grid(False)
    ax.view_init(elev=30, azim=45)
    
    # Log sigmoid(f) image to WandB
    wandb.log({"Sigmoid_f_Image": wandb.Image(fig)})
    # save_path = os.path.join(f"testing1/sigmoidf_{timestamp}.png")
    # plt.savefig(save_path)
    plt.close()
    
    return values,ackley_values, f
################################################## Generate Reward in RKHS and f as diff in rewards #######################################
def generate_preference_RKHS(grid_size=100,alpha_gp=0.05,length_scale=0.1,n_samples=10,base_kernel=None,smoothness=1.5,random_state=None): 
    # Set seed for reproducibility if provided
    # Generate 1D values
    values = np.linspace(0, 1, grid_size)
    if base_kernel == "Matern":
        kernel = Matern(length_scale=length_scale, nu=smoothness, length_scale_bounds="fixed")
      
    elif base_kernel =="RBF":
        kernel = RBF(length_scale=length_scale, length_scale_bounds="fixed")

    
    # Initialize the GaussianProcessRegressor
    gp = GaussianProcessRegressor(kernel=kernel,random_state=random_state)

    # # Number of sample points for training
    # n_samples = 10

    # Generate sample points in the 1D domain
    values_samples = np.linspace(0, 1, n_samples)
    X_gp = values_samples.reshape(-1, 1)
    #print('X_gp',X_gp)
    # Sample the function values at these points
    #y = gp.sample_y(X_gp, random_state=random_state).ravel()
    y = np.array([-0.5, 0, 0.5, 1, 0.7, 0.2, 0.2, 0.5, 0.95, 0.7])
    #print('y',y)

    # Initialize and fit the GaussianProcessRegressor
    gpr = GaussianProcessRegressor(kernel=kernel, optimizer=None, alpha=alpha_gp, random_state=random_state)
    gpr.fit(X_gp, y)
    # Generate a dense grid of values for prediction
    X_full = np.linspace(0, 1, grid_size).reshape(-1, 1)
    #print('X_full',X_full)

    # Predict mean for the dense grid
    y_pred = gpr.predict(X_full)
    Reward_function=y_pred

    # Plot the results
    plt.figure(figsize=(10, 6))



    # Plot the predicted mean
    plt.plot(X_full, y_pred, 'b-')

    plt.xlabel(r"$x$", fontsize=20)
    plt.ylabel(r"$f(x)$",fontsize=20)

    plt.legend()
    plt.grid(False)
     # Log R image to WandB
    wandb.log({"R Image": wandb.Image(plt)})


    #save_path = os.path.join(f"testing/R_{timestamp}.png")
    #plt.savefig(save_path)
    plt.close()
  
    # print('X_full',X_full)
    # print('R',y_pred)
    # Initialize the matrix f with zeros
    f = np.zeros((grid_size, grid_size))

    # Compute f[i, j] = R[i] - R[j]
    for i in range(grid_size):
        for j in range(grid_size):
            f[i, j] = Reward_function[i] - Reward_function[j]
    f=f*5
    # value_at_x1_0_x2_1 = f[0, grid_size - 1]

    # print("Value at x1=0, x2=1:", value_at_x1_0_x2_1)
    #print('f',f)
    #f = scale_to_range(f, -3, 3)

    # Plot f
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create meshgrid for plotting
    action_mesh, state_mesh = np.meshgrid(values, values)
    # print('f',f)

    # Plot the surface
    surf = ax.plot_surface(state_mesh, action_mesh, f, cmap='viridis')
    # Add a dot at the specified point
    # x1_dot = 0
    # x2_dot = 1
    # z_dot = f[x1_dot,99]  # Calculate the corresponding z value
    # ax.scatter(x1_dot, x2_dot, z_dot, color='red', s=50)

    # Add color bar which maps values to colors
    fig.colorbar(surf, shrink=0.5, aspect=5)

    # Set axis labels
    ax.set_xlabel(r"$x$", fontsize=20)
    ax.set_ylabel(r"$x^{'}$", fontsize=20)
    ax.set_zlabel(r"$h(x,x^{'})$", fontsize=20)
    ax.grid(False)

    # Customize the view angle
    ax.view_init(elev=30, azim=45)

    
    # Log f image to WandB
    wandb.log({"f Image": wandb.Image(fig)})

    # Save the figure
    # save_path = os.path.join(f"testing/f_{timestamp}.png")
    # plt.savefig(save_path)
    plt.close()

        # Calculate sigmoid of f
    sigmoid_f = sigmoid(f)

    #Plotting sigmoid(f)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(state_mesh, action_mesh, sigmoid_f, cmap='viridis')
    fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.set_xlabel(r"$x$", fontsize=20)
    ax.set_ylabel(r"$x^{'}$", fontsize=20)
    ax.set_zlabel(r"$\mu(h(x, x^{'}))$", fontsize=20)
    ax.zaxis.label.set_rotation(180)
    ax.grid(False)
    ax.view_init(elev=30, azim=45)
    # ax.text2D(0.05, 0.95, f'Alpha: {alpha_gp}', transform=ax.transAxes, fontsize=14, color='red')

    wandb.log({"Sigmoid_f_Image": wandb.Image(fig)})
    plt.close()
    # save_path = os.path.join(f"testing/sigmoid_f_{timestamp}.png")
    # plt.savefig(save_path)
    return values, Reward_function, f


####################################################### Part 2: sigmoid(f) ###################################################################

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


################################################## Part 3: All functions ######################################################################


class DuelingKernel2(Kernel):
    def __init__(self, base_kernel=None, length_scale=0.1, smoothness=1.5):
        self.length_scale = length_scale
        self.length_scale_bounds= "fixed"
        self.base_kernel=base_kernel
        self.smoothness=smoothness
        if self.base_kernel == "Matern":
            self.kernel = Matern(length_scale=length_scale, nu=smoothness)
      
        elif self.base_kernel =="RBF":
            self.kernel = RBF(length_scale=length_scale)
            
        
    def __call__(self, X, Y=None, eval_gradient=False):
        return self.dueling_kernel(X, Y,eval_gradient=eval_gradient)
    
    def dueling_kernel(self, X, Y=None,eval_gradient=False):
        if Y is None:
            Y = X
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
       
          # Split X and Y into their components
        X1, X2 = X[:, 0:1], X[:, 1:2]
        Y1, Y2 = Y[:, 0:1], Y[:, 1:2]

            # Compute the RBF kernel components
        K_x1_x1p = self.kernel(X1, Y1)  # k(x1, x1')
        K_x2_x2p = self.kernel(X2, Y2)  # k(x2, x2')
        K_x1_x2p = self.kernel(X1, Y2)  # k(x1, x2')
        K_x2_x1p = self.kernel(X2, Y1)  # k(x2, x1')
        
        # Combine the kernel components to form the custom kernel
        K = K_x1_x1p + K_x2_x2p - K_x1_x2p - K_x2_x1p
        if X.shape == Y.shape:
            K += np.eye(X.shape[0]) * 1e-6
       
        
        if eval_gradient:
            _, grad_x1_x1p = self.kernel(X1, Y1, eval_gradient=True)
            _, grad_x2_x2p = self.kernel(X2, Y2, eval_gradient=True)
            _, grad_x1_x2p = self.kernel(X1, Y2, eval_gradient=True)
            _, grad_x2_x1p = self.kernel(X2, Y1, eval_gradient=True)
            
            gradient = grad_x1_x1p + grad_x2_x2p - grad_x1_x2p - grad_x2_x1p
            return K, gradient
           
        else:
    
            return K
     
        
              
    def diag(self, X):
        return np.diag(self.__call__(X))
    
    def is_stationary(self):
        return True
  
    
# Loss function
def loss_function(alpha, K, y, lambda_reg):
    alpha = alpha.reshape(-1, 1)
    # print('alpha',alpha)
    K_alpha = K @ alpha
    #print('K_alpha',K_alpha)
    sig = sigmoid(K_alpha)
    #print('sigmoid k_alpha',sig.shape)
    #print('y',y.shape)
      # Reshape y to match the shape of K_alpha
    y_reshaped = y.reshape(-1, 1)
    #print('y_reshaped',y_reshaped)
    # Clip the sigmoid outputs to avoid log(0)
    eps = 1e-10
    sig = np.clip(sig, eps, 1 - eps)
    # Calculate individual losses (element-wise)
    loss_indiv = - (y_reshaped * np.log(sig) + (1 - y_reshaped) * np.log(1 - sig))
     # Normalize the loss by the number of samples
    N = y.shape[0]
    #print('Loss individual shape:', loss_indiv)
    
    # Calculate total loss
    loss = (1 / N) * np.sum(loss_indiv) + (lambda_reg / 2) * np.sum(alpha**2)
    #print('Total loss:', loss)

    # print('loss indiv',y*np.log(sig) + (1 - y) * np.log(1 - sig))
    # loss = -np.sum(y * np.log(sig) + (1 - y) * np.log(1 - sig)) + (lambda_reg / 2) * np.sum(alpha**2)
    # print('loss',loss)
    return loss


# Gradient of the loss function
def gradient_loss_function(alpha, K, y, lambda_reg):
    alpha = alpha.reshape(-1, 1)
    #print('alpha',alpha.shape)
    K_alpha = K @ alpha
    #print('k_alpha',K_alpha.shape)
    sig = sigmoid(K_alpha)
    eps = 1e-10
    sig = np.clip(sig, eps, 1 - eps)
    # Normalize the gradient by the number of samples
    N = y.shape[0]
    grad = (1 / N)* K.T @ (sig - y.reshape(-1, 1)) + lambda_reg * alpha #not sure with or without transpose
    #print('grad',grad.shape)
    return grad.flatten()



# Define the function f_t using the optimal alpha
def f_t(x, X_train, optimal_alpha, kernel):
    k_t = kernel(X_train, np.array([x])).flatten()
    return np.dot(optimal_alpha, k_t)


def predict_f(dataset, values,grid_size,kernel,lambda_reg=0.05,learning_rate=0.1,n_iterations_GD=200000,lr_decay=0,filename=None):
        
    if filename is not None:
        file = open(filename,"a")
    else:
        file=None
    # New part for logistic regression with kernel trick
    X_train = dataset[:, :2]
    #print('X_train',X_train)
    y_train = dataset[:, 2]
    #print('y_train',y_train)
    initial_learning_rate = learning_rate
    
    tolerance = 1e-6  # Threshold for stopping
    grad_norm_threshold = 1e-6
    previous_loss = float('inf')


    # Define the kernel using scikit-learn's RBF kernel 
    #kernel = RBF(length_scale=length_scale)

    # Compute the kernel matrix using scikit-learn's pairwise_kernels function
    #K = pairwise_kernels(X_train, metric=kernel)
    # Compute the kernel matrix using the custom dueling kernel function
    K = kernel(X_train)
    #print('kernel matrix',K)


    # Initialize alpha with zeros
    alpha = np.zeros(X_train.shape[0])
    decay_rate = 0.1
    # Gradient descent
    for iteration in range(n_iterations_GD):
        if lr_decay != 0:
            #print('learning rate decay 0')
            learning_rate = initial_learning_rate / (1 + decay_rate * iteration)
        else:
            #print('learning rate decay is 0')
            learning_rate=initial_learning_rate
        grad = gradient_loss_function(alpha, K, y_train, lambda_reg)
    
        alpha -= learning_rate * grad
    

        
        current_loss = loss_function(alpha, K, y_train, lambda_reg)
        
        # Compute the norm of the gradient
        grad_norm = np.linalg.norm(grad)

    # Periodic logging of loss
        if iteration % 100000 == 0:
            output_string = f"Iteration {iteration}, Loss: {current_loss}, param: {alpha}\n"
            # print(output_string)
            if file is not None:
                file.write(output_string)
           
            #print(f"Iteration {iteration}, Loss: {current_loss}, param: {alpha}")
            
        # # # Convergence check
        if abs(previous_loss - current_loss) < tolerance:
            #print(f"Convergence in loss reached at iteration {iteration}")
            break
        if grad_norm < grad_norm_threshold:
        # print(f"Convergence due to small gradient norm at iteration {iteration}")
            break

    # Update previous_loss
        previous_loss = current_loss
        
    # Optimal alpha
    optimal_alpha = alpha
    if file is not None:
        file.write(f'Final loss: {current_loss}\n')
    # print('final loss',current_loss)
    # Testing f_t on the grid
    f_values = np.zeros((grid_size, grid_size))
    for i, x1 in enumerate(values):
        for j, x2 in enumerate(values):
            f_values[i, j] = f_t([x1, x2], X_train, optimal_alpha, kernel)
    if file is not None:
        file.write(f'f_values: {f_values}\n')
    # np.savetxt(file, f_values, delimiter=",")
    # print('f_values',f_values)

    return f_values, current_loss

def update_sigma_D(dataset, kernel, alpha_gp, values):
    #kernel= RBF(length_scale=0.1)
    X = dataset[:, :2]
    #print('X',X.shape)
    y = dataset[:, 2]
    #print('y',y.shape)
    #print('kernel',kernel)
    #print('alpha',alpha_gp)
    #gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=40,optimizer="fmin_l_bfgs_b",alpha=alpha_gp) # return optimizer=None
    gp = GaussianProcessRegressor(kernel=kernel,optimizer=None,alpha=alpha_gp) 
    gp.fit(X, y)
    X_grid = np.array(list(product(values, repeat=2)))
    #print('X_grid',X_grid.shape)
    _, y_std = gp.predict(X_grid, return_std=True)
    #print('y_std',y_std)
    return y_std.reshape(len(values), len(values))

def update_M_t(f_values, sigma_D, beta, values,threshold=0.5):
    M_t = []
    for i, x in enumerate(values):
        all_conditions_met = True
        for j, x_prime in enumerate(values):
            if sigmoid(f_values[i, j]) + beta * sigma_D[i, j] < threshold:
                all_conditions_met = False
                break
        if all_conditions_met:
            M_t.append(x)
    return M_t
    
def update_M_t_previous(f_values, sigma_D, beta, values, Mt_prev, threshold=0.5):
    M_t = []
    #prev_indices = [values.index(x) for x in M_t_prev]  # Convert previous M_t to indices
    prev_indices = [np.where(values == x)[0][0] for x in Mt_prev]  # Using np.where
    #print('prev_indices',prev_indices)
    for i in prev_indices:
        x = values[i]
        all_conditions_met = True
        
        for j in prev_indices:
            x_prime = values[j]
            if sigmoid(f_values[i, j]) + beta * sigma_D[i, j] < threshold:
                all_conditions_met = False
                break
        
        if all_conditions_met:
            M_t.append(x)
    
    return M_t

def select_pair(values, f_values, sigma_D, beta, M_t):
    def objective_leader(x, x_prime):
        i = np.searchsorted(values, x[0])
        j = np.searchsorted(values, x_prime[0])
        return sigmoid(f_values[i, j]) - beta * sigma_D[i, j]

    def objective_follower(x_prime, x):
        i = np.searchsorted(values, x[0])
        j = np.searchsorted(values, x_prime[0])
        return sigmoid(f_values[i, j]) - beta * sigma_D[i, j]

    best_pair = None
    
    max_value = -np.inf

    for x1 in M_t:
        #('x:', x1)
        
        # Brute force over all x2 in M_t
        best_x_prime = None
        min_follower_value = np.inf

        for x2 in M_t:
            follower_value = objective_follower([x2], [x1])
            if follower_value < min_follower_value:
                min_follower_value = follower_value
                best_x_prime = [x2]
        #print('best_x_prime',best_x_prime)
        if best_x_prime is not None:
            value = objective_leader([x1], best_x_prime)
            # print('x_prime:', best_x_prime, 'value:', value)
            if value > max_value:
                max_value = value
                best_pair = ([x1], best_x_prime)

    #print('best pair',best_pair)    
    return best_pair


def find_most_preferred_action(Reward_function,values):
        # Find the index of the maximum value in the reward function
    max_index = np.argmax(Reward_function)
    #print('max_index',max_index)
    
    # Return the input value corresponding to the maximum reward
    x_star = values[max_index]
    #print('x_star',x_star)
    
    return x_star

# def find_x_star_predicted(f_values, values):
#     # Apply the sigmoid function to f_values
#     sigmoid_f_values = sigmoid(f_values)
    
#     # Find the indices of the maximum value in the sigmoid_f_values
#     max_index = np.argmax(sigmoid_f_values)
#     i, j = np.unravel_index(max_index, sigmoid_f_values.shape)
    
#     # Map the indices back to the original values
#     x_star_predicted = (values[i], values[j])
    
#     return x_star_predicted

# Function to find x_star_predicted
def find_x_star_predicted(f_values, values):
 
    # Initialize best_x_star and max_count
    best_x_star = None
    max_count = -1
    
    # Iterate through all x_star candidates
    for i, x_star in enumerate(values):
        #print('x_star',x_star)
        # Count how many comparisons x_star wins
        count = 0
        
        # Compare x_star with every other value
        for j, x in enumerate(values):
            if i != j:
                # Calculate sigmoid of f(x_star, x)
                p = sigmoid(f_values[i, j])
                
                # Check if p > 0.5
                if p > 0.5:
                    count += 1
        
        # Update best_x_star if count is greater than max_count
        #print('count',count)
        if count > max_count:
            max_count = count
            best_x_star = x_star
    
    return best_x_star

# def find_x_star_ackley(f_values, values):
#     # Initialize best_x_star and max_count
#     best_x_star = None
#     max_count = -1
    
#     # Iterate through all x_star candidates
#     for i, x_star in enumerate(values):
#         count = 0
        
#         # Compare x_star with every other value
#         for j, x in enumerate(values):
#             if i != j:
#                 # Calculate sigmoid of f(x_star, x)
#                 p = sigmoid(f_values[i, j])
                
#                 # Check if p > 0.5
#                 if p > 0.5:
#                     count += 1
        
#         # Update best_x_star if count is greater than max_count
#         if count > max_count:
#             max_count = count
#             best_x_star = x_star
    
#     return best_x_star






def compute_regret(x_star, x, x_prime, f, values):
    i_star = np.searchsorted(values, x_star)
    i = np.searchsorted(values, x)
    j = np.searchsorted(values, x_prime)
    mu_f_x_star_x = sigmoid(f[i_star, i])
    mu_f_x_star_x_prime = sigmoid(f[i_star, j])
    regret = (mu_f_x_star_x + mu_f_x_star_x_prime - 1) / 2
    return regret

######################################################################
def parse_arguments():
    parser = argparse.ArgumentParser(description='Run the dueling bandits experiment.')

    parser.add_argument('--alpha_gp', type=float, default=0.05,
                        help='The alpha parameter for Gaussian Process Regression.')
    parser.add_argument('--length_scale', type=float, default=0.1,
                        help='The length scale for the kernel.')
    parser.add_argument('--grid_size', type=int, default=100,
                        help='The size of the grid.')
    parser.add_argument('--lambda_reg', type=float, default=0.05,
                        help='The regularization parameter for loss function.')
    parser.add_argument('--beta', type=float, default=1,
                        help='The beta parameter.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='The learning rate for gradient descent.')
    parser.add_argument('--n_samples', type=int, default=10,
                        help='The number of samples.')
    parser.add_argument('--n_iterations_GD', type=int, default=200000,
                        help='The number of iterations for gradient descent.')
    parser.add_argument('--n_iterations', type=int, default=300,
                        help='The number of iterations for the main loop.')
    parser.add_argument('--n_runs', type=int, default=10,
                        help='The number of runs.')
    parser.add_argument("--algo", type=str, default="Max_Min_LCB", help="Name of the model")
    parser.add_argument("--lr_decay", type=int, default= 0, help= "learning rate decaying or not")
    # parser.add_argument('--log_freq', type=int, default=10,
    #                     help='The frequency of logging.')
    parser.add_argument("--kernel", type=str, default="RBF", help="Kernel type")
    parser.add_argument("--smoothness", type=float, default=1, help="Smoothness of the kernel")
    parser.add_argument("--seed", type=int, default=0, help="random state for the preference function generation")
    parser.add_argument("--preference_function", type=str, default= "RKHS", help= "Test preference function in RKHS or ackley function from the optimization literature")
    parser.add_argument("--enable_logging", type= int, default = 0, help ='log txt files')
    parser.add_argument("--append_identical_pairs", type= int, default = 0, help ='append identical action pairs')
    parser.add_argument("--repeat_identical_pairs", type= int, default = 1, help ='Number of identical pairs to append in the dataset')  
    args = parser.parse_args()
    return args
def Max_Min_LCB(args, values, Reward_function, f,timestamp):
    dataset = np.empty((0, 3))
    M_t = values.tolist()  # Start with all possible x values
    regret_list = []
    
    dueling_kernel_instance = DuelingKernel2(base_kernel=args.kernel,length_scale=args.length_scale, smoothness=args.smoothness)
    # Get the current timestamp
    identical_pairs_count = 0  # Counter for identical pairs
    total_pairs_count = 0  # Total number of pairs selected
    if args.enable_logging:
        #print('yes')
        log_filename = f"MaxMinLCB_{args.kernel}_{args.learning_rate}_{args.lr_decay}_{timestamp}.txt"
    else:
        #print('no')
        log_filename = None

    
    # Conditionally open the file if logging is enabled
    file = open(log_filename, "a") if log_filename else None
    
    try:

        for t in range(args.n_iterations):
                print('t',t)
                if len(dataset) == 0:
                    f_values = np.zeros((args.grid_size, args.grid_size))  # Random initialization
                    sigma_D = np.zeros((args.grid_size, args.grid_size))  # Random initialization
                else:
                    f_values, loss = predict_f(dataset, values, args.grid_size, dueling_kernel_instance, lambda_reg=args.lambda_reg, learning_rate=args.learning_rate, n_iterations_GD=args.n_iterations_GD, lr_decay=args.lr_decay,filename=log_filename)
                    wandb.log({
                        "loss": loss  
                    })
                    sigma_D = update_sigma_D(dataset, dueling_kernel_instance, alpha_gp=args.alpha_gp, values=values)
                    M_t = update_M_t(f_values, sigma_D, beta=args.beta, values=values)
                
                    
                
                pair = select_pair(values, f_values, sigma_D, beta=args.beta, M_t=M_t)
                output_string = f"Iteration {t}, Selected Pair: {pair}\n"
                print(output_string)
                if file:
                    file.write(output_string)
                
                if pair:
                    x, x_prime = pair
                    if x == x_prime:
                        identical_pairs_count += 1
                    total_pairs_count += 1
                    i = np.searchsorted(values, x[0])
                    j = np.searchsorted(values, x_prime[0])
                    p = sigmoid(f[i, j])
                    y = np.random.binomial(1, p)
                    dataset = np.vstack((dataset, [x[0], x_prime[0], y]))
                    x_star = find_most_preferred_action(Reward_function, values)
                    regret = compute_regret(x_star, x[0], x_prime[0], f, values)
                    regret_list.append(regret)
                    wandb.log({
                        "iteration": t,
                        "Regret": regret
                    })
                    
    finally:
        # Close the file if it was opened
        if file:
            file.close()
    if total_pairs_count > 0:
        identical_percentage = (identical_pairs_count / total_pairs_count) * 100
        wandb.log({"Identical Pairs Percentage": identical_percentage})
    return regret_list, M_t

def BOHF_SimpleRegret(args, values, Reward_function, f,timestamp):
    dataset = np.empty((0, 3))
    
    dueling_kernel_instance = DuelingKernel2(base_kernel=args.kernel,length_scale=args.length_scale, smoothness=args.smoothness)
    if args.enable_logging:
        #print('yes')
        log_filename = f"BOHF_SimpleRegret_{args.kernel}_{args.learning_rate}_{args.lr_decay}_{timestamp}.txt"
    else:
        #print('no')
        log_filename = None

    
    # Conditionally open the file if logging is enabled
    file = open(log_filename, "a") if log_filename else None
    
    try:

        for t in range(args.n_iterations):
                if len(dataset) == 0:
                    sigma_D = np.zeros((args.grid_size, args.grid_size))  # Random initialization
                else:
                
                    sigma_D = update_sigma_D(dataset, dueling_kernel_instance, alpha_gp=args.alpha_gp, values=values)
                    #M_t = update_M_t(f_values, sigma_D, beta=args.beta, values=values)

                # Find indices that maximize the standard deviation
                #print('sigma_D',sigma_D)
                max_indices = np.unravel_index(np.argmax(sigma_D, axis=None), sigma_D.shape)
                i, j = max_indices
                #print('i ',i,' j',j)
                pair = (values[i], values[j])
                output_string = f"Iteration {t}, Selected Pair: {pair}\n"
                print(output_string)
                if file:
                    file.write(output_string)
                #print('pair',pair)
                
                if pair:
                    x, x_prime = pair
                    p = sigmoid(f[i, j])
                    y = np.random.binomial(1, p)
                    dataset = np.vstack((dataset, [x, x_prime, y]))
                    
                    #print('actual x star',x_star)
                    #regret = compute_regret(x_star, x[0], x_prime[0], f, values)
                    #regret_list.append(regret)
                    #wandb.log({
                    #    "iteration": t,
                    #    "Regret": regret
        
                    #})
 
        f_values, loss = predict_f(dataset, values, args.grid_size, dueling_kernel_instance, lambda_reg=args.lambda_reg, learning_rate=args.learning_rate, n_iterations_GD=args.n_iterations_GD,lr_decay= args.lr_decay,filename=log_filename)
        wandb.log({
        "loss": loss  
        })
        x_star = find_most_preferred_action(Reward_function, values)
        best_x_star=find_x_star_predicted(f_values, values)
        #regret = compute_regret(x_star, x, x_prime, f, values)
        regret = compute_regret(x_star, best_x_star, best_x_star, f, values)
        #print('best_x_star predicted',best_x_star)
        #print('regret',regret)
        
    finally:
    # Close the file if it was opened
        if file:
            file.close()
    return best_x_star,regret,x_star

def BOHF(args, values, Reward_function, f,timestamp):
    M_t = values.tolist()  # Start with all possible x values
    regret_list = []
    dueling_kernel_instance = DuelingKernel2(base_kernel=args.kernel,length_scale=args.length_scale, smoothness= args.smoothness)
    N=1
    t=0
    T=args.n_iterations
    identical_pairs_count = 0  # Counter for identical pairs
    total_pairs_count = 0  # Total number of pairs selected
    if args.enable_logging:
        #print('yes')
        log_filename = f"BOHF_{args.kernel}_{args.learning_rate}_{args.lr_decay}_{timestamp}.txt"
    else:
        #print('no')
        log_filename = None

    
    # Conditionally open the file if logging is enabled
    file = open(log_filename, "a") if log_filename else None
    
    try:
        while True:
            print('t',t)
            N= int(np.ceil(np.sqrt(T* N)))
            print('N',N)
            dataset_round = np.empty((0, 3))
            # Generate all identical pairs (x, x) and append to dataset_round
            if args.append_identical_pairs:
                # print('appending identical pairs')
                for x in values:
                    #print('x',x)
                    i = np.where(values == x)[0][0]
                    #print('i',i)
                    for _ in range(args.repeat_identical_pairs):
                        p = sigmoid(f[i, i])  # Compute the probability using f[i, i]
                        y = np.random.binomial(1, p)  # Sample from a binomial distribution
                        dataset_round = np.vstack((dataset_round, [x, x, y]))
            for n in range(N):
                # print('dataset_round',dataset_round)
                if len(dataset_round) == 0:
                    #f_values = np.zeros((args.grid_size, args.grid_size))  # Random initialization
                    sigma_D = np.zeros((args.grid_size, args.grid_size))  # Random initialization
                else:
                
                    sigma_D = update_sigma_D(dataset_round, dueling_kernel_instance, alpha_gp=args.alpha_gp, values=values)
                    # M_t = update_M_t(f_values, sigma_D, beta=args.beta, values=values) ## we do not update after each pair, but after each whole round
                    # Restrict max_indices search to current M_t
                

                #M_t_indices = [values.index(x) for x in M_t]   
                M_t_indices = [np.where(values == x)[0][0] for x in M_t]

                #print('M_t_indices',M_t_indices) 
                #print('sigma_D',sigma_D)
                sigma_D_subset = sigma_D[np.ix_(M_t_indices, M_t_indices)]
                #print('sigma_D_subset',sigma_D_subset )
                max_indices_subset = np.unravel_index(np.argmax(sigma_D_subset, axis=None), sigma_D_subset.shape)
                #print(' max_indices_subset', max_indices_subset)
                
                i, j = M_t_indices[max_indices_subset[0]], M_t_indices[max_indices_subset[1]]
                #print('i ',i,' j',j)
                pair = (values[i], values[j])
                output_string = f"Iteration {t}, Selected Pair: {pair}\n"
                print(output_string)
                if file:
                    file.write(output_string)
                #print('pair',pair)
                if values[i] == values[j]:
                    identical_pairs_count += 1
                total_pairs_count += 1
                x, x_prime = pair

            
                p = sigmoid(f[i, j])
                y = np.random.binomial(1, p)
                dataset_round = np.vstack((dataset_round, [x, x_prime, y]))
                x_star = find_most_preferred_action(Reward_function, values)
                regret = compute_regret(x_star, x, x_prime, f, values)
                regret_list.append(regret)
                wandb.log({
                    "iteration": t,
                    "Regret": regret
                })
                t+=1
                print('t',t)
                if t>=T:
                    if total_pairs_count > 0:
                        final_identical_percentage = (identical_pairs_count / total_pairs_count) * 100
                        wandb.log({
                            "Final Identical Pairs Percentage": final_identical_percentage
                        })
                    print('regret',regret_list)
                    return regret_list, M_t
            
            sigma_D = update_sigma_D(dataset_round, dueling_kernel_instance, alpha_gp=args.alpha_gp, values=values)
            f_values, loss = predict_f(dataset_round, values, args.grid_size, dueling_kernel_instance, lambda_reg=args.lambda_reg, learning_rate=args.learning_rate, n_iterations_GD=args.n_iterations_GD, lr_decay= args.lr_decay,filename=log_filename)
            wandb.log({"loss": loss})
            M_t = update_M_t_previous(f_values, sigma_D, beta=args.beta, values=values, Mt_prev=M_t) ## we do not update after each pair, but after each whole round, change how u update Mt
    finally:
        # Close the file if it was opened
        if file:
            file.close()
            
    return regret_list, M_t
    
        

def main():
    args = parse_arguments()
    
    # Prepare for storing regrets across runs
    all_regret_lists = []
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"experiment/{args.algo}_{args.kernel}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
    
    
    for run in range(args.n_runs):

        # Initialize wandb for each run
        wandb.init(project="BOHF_identical_pairs2", reinit=True, settings=wandb.Settings(start_method="thread"))#, settings=wandb.Settings(start_method="thread"))
        
        # Log run-specific parameters
        wandb.run.summary["algo"] = args.algo
        wandb.run.summary["run_number"] = run
        wandb.run.summary["grid_size"] = args.grid_size
        wandb.run.summary["alpha_gp"] = args.alpha_gp
        wandb.run.summary["length_scale"] = args.length_scale
        wandb.run.summary["beta"] = args.beta
        wandb.run.summary["lambda_reg"] = args.lambda_reg
        wandb.run.summary["learning_rate"] = args.learning_rate
        wandb.run.summary["num_iterations"] = args.n_iterations
        wandb.run.summary["n_iterations_GD"] = args.n_iterations_GD
        wandb.run.summary["save_dir"]= save_dir
        wandb.run.summary["lr_decay"]= args.lr_decay
        wandb.run.summary["kernel"] = args.kernel
        wandb.run.summary["smoothness"] = args.smoothness
        wandb.run.summary["seed"] = args.seed
        wandb.run.summary["preference_function"] = args.preference_function
        wandb.run.summary["append_identical_pairs"]=args.append_identical_pairs
        wandb.run.summary["repeat_identical_pairs"]=args.repeat_identical_pairs

        # Generate the preference function and reward
        if args.preference_function == "RKHS":
            values, Reward_function, f = generate_preference_RKHS(grid_size=args.grid_size, alpha_gp=args.alpha_gp, length_scale=args.length_scale, n_samples=args.n_samples, base_kernel= args.kernel, smoothness= args.smoothness,random_state= args.seed) #alpha here was args.alpha_gp
        elif args.preference_function == "ackley":
            values,Reward_function,f= generate_ackley(grid_size=args.grid_size)
        if args.algo == "Max_Min_LCB":
            regret_list, M_t=Max_Min_LCB(args,values,Reward_function,f,timestamp)
            #print('M_t',M_t)
            # wandb.log({"Final_M_t": M_t})
        
            # After the loop: Store the regret list for this run
            if regret_list:
                all_regret_lists.append(regret_list)

            
        # Plot the regret over iterations
            plt.figure(figsize=(10, 6))
            plt.plot(regret_list, label='Regret')
            plt.xlabel('Iteration')
            plt.ylabel('Regret')
            plt.title('Regret over Iterations')
            plt.legend()
            wandb.log({"Regret Plot": wandb.Image(plt)})
            plt.close()

            # Finish the current run
            wandb.finish()
        elif args.algo =="BOHF_SimpleRegret":
            best_x_star,regret,x_star = BOHF_SimpleRegret(args, values, Reward_function, f,timestamp)
            # Log results to wandb
            wandb.log({
                'X* predicted': best_x_star,
                'Regret': regret,
                'X*': x_star
            })
        elif args.algo == "BOHF":
            regret_list, M_t=BOHF(args,values,Reward_function,f,timestamp)
            #print('M_t',M_t)
            # wandb.log({"Final_M_t": M_t})
        
            # After the loop: Store the regret list for this run
            if regret_list:
                all_regret_lists.append(regret_list)

            
        # Plot the regret over iterations
            plt.figure(figsize=(10, 6))
            plt.plot(regret_list, label='Regret')
            plt.xlabel('Iteration')
            plt.ylabel('Regret')
            plt.title('Regret over Iterations')
            plt.legend()
            wandb.log({"Regret Plot": wandb.Image(plt)})
            plt.close()

            # Finish the current run
            wandb.finish()
    
    # After all runs: Compute and log the average of regrets at each iteration across all runs
    if all_regret_lists:
        # Convert to numpy array for easier manipulation
        # Convert to numpy array if it's not already
        all_regret_arrays = np.array(all_regret_lists)
        np.save(os.path.join(save_dir, 'all_regret_arrays.npy'), all_regret_arrays)

        # Number of runs and iterations
        n_runs, n_iterations = all_regret_arrays.shape

        # Calculate mean regret across runs at each iteration
        mean_regret = np.mean(all_regret_arrays, axis=0)

        # Calculate standard deviation of regrets at each iteration
        std_dev_regret = np.std(all_regret_arrays, axis=0)

        # Calculate standard error of regrets at each iteration
        std_error_regret = std_dev_regret / np.sqrt(n_runs)

        # Plotting mean regret with standard error as error bars
        plt.figure(figsize=(10, 6))
        plt.plot(mean_regret, label='Mean Regret')
        plt.fill_between(range(n_iterations),
                        mean_regret - std_error_regret,
                        mean_regret + std_error_regret,
                        color='b', alpha=0.2, label='Standard Error')
        plt.xlabel('Iteration')
        plt.ylabel('Regret')
        plt.title('Mean Regret with Standard Error Over Iterations')
        plt.legend()
        plt.grid(True)
        # Save the figure
        filename = "Mean_regret_runs.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path)

if __name__ == "__main__":
    main()