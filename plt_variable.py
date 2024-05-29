import csv
import numpy as np
import matplotlib.pyplot as plt
import os

# Define path and filename
csv_path = 'losses_and_spreads_variable_trade_sizeVAR_MULTIPLE1.csv'
filename = os.path.basename(csv_path).split(".")[0]

# Create a directory to save plots if it doesn't exist
output_dir = filename
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# Read CSV and process data
data = {}
columns = ['alpha', 'sigma', 'loss_rl', 'spread_rl', 'mid_dev_rl', 'loss_compare', 'spread_compare', 'mid_dev_compare', 'loss_bayes', 'spread_bayes', 'mid_dev_bayes']
for col in columns:
    data[col] = []

# Open the CSV file
with open(csv_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)  # Skip the header if there is one
    for row in reader:
        data['alpha'].append(float(row[0]))
        data['sigma'].append(float(row[1]))
        data['loss_rl'].append(float(row[4]))
        data['spread_rl'].append(float(row[5]))
        data['mid_dev_rl'].append(float(row[9])/float(row[17]))
        data['loss_compare'].append(float(row[12]))
        data['spread_compare'].append(float(row[13]))
        data['mid_dev_compare'].append(float(row[17])/float(row[17]))
        data['loss_bayes'].append(float(row[20]))
        data['spread_bayes'].append(float(row[21]))
        data['mid_dev_bayes'].append(float(row[25])/float(row[17]))

# Define plotting function
def plot_function(x, y, xlabel, ylabel, title_suffix, filename_suffix):
    metrics = ['loss', 'spread', 'mid_dev']
    for metric in metrics:
        #plt.figure(figsize=(10, 5))
        plt.plot(x, y[metric + '_rl'], label='AKF (Ours)', marker='o')
        plt.plot(x, y[metric + '_bayes'], label='KF (Ours)', marker='o')
        plt.plot(x, y[metric + '_compare'], label='Uniswap', marker='o')
        plt.axhline(0, color='gray', linewidth=0.8)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.xlabel(xlabel)
        ylabel_str = "Monetary Loss" if metric == "loss" else ("Spread" if metric == "spread" else "Price Deviation")
        plt.ylabel(ylabel_str)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{metric}_vs_{filename_suffix}.pdf")
        plt.close()

# Prepare data for plotting
def prepare_data_for_plot(group_by, against):
    unique_values = np.unique(data[group_by])
    plots_data = []
    for value in unique_values:
        indices = np.where(data[group_by] == value)
        plot_data = {key: np.array(data[key])[indices] for key in data if key != group_by}
        plot_data[group_by] = np.array(data[against])[indices]
        plots_data.append((value, plot_data))
    return plots_data


#print(data)
#Generate plots
alpha_plots = prepare_data_for_plot('alpha', 'sigma')
for alpha, plot_data in alpha_plots:
    plot_function(plot_data['sigma'], plot_data, "Volatility", "Value", f"Volatility for Trader Noise={alpha}", f"sigma_for_alpha_{alpha}")

sigma_plots = prepare_data_for_plot('sigma', 'alpha')
for sigma, plot_data in sigma_plots:
    plot_function(plot_data['alpha'], plot_data, "Volatility of Volatility", "Value", f"Trader Noise for Volatility={sigma}", f"alpha_for_sigma_{sigma}")

from mpl_toolkits.mplot3d import Axes3D
metrics = ['loss', 'spread', 'mid_dev']
# # Function to compute average over groups
def compute_averages(group_by, data):
    unique_values = np.unique(data[group_by])
    averaged_data = {metric + i : [] for metric in metrics for i in ['_rl','_bayes','_compare']}
    averaged_data[group_by] = unique_values
    for value in unique_values:
        indices = np.where(data[group_by] == value)
        for metric in metrics:
            averaged_data[metric + '_rl'].append(np.mean(np.array(data[metric + '_rl'])[indices]))
            averaged_data[metric + '_bayes'].append(np.mean(np.array(data[metric + '_bayes'])[indices]))
            averaged_data[metric + '_compare'].append(np.mean(np.array(data[metric + '_compare'])[indices]))

    return averaged_data

# # Function to plot 3D surface
# def plot_3d(x, y, z, xlabel, ylabel, zlabel, title):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     X, Y = np.meshgrid(x, y)
#     Z = np.array(z).reshape(len(y), len(x))
#     surf = ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=False)
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
#     ax.set_zlabel(zlabel)
#     plt.title(title)
#     plt.show()

# # Usage example for averaging and plotting
alpha_avg_data = compute_averages('alpha', data)
sigma_avg_data = compute_averages('sigma', data)
print(alpha_avg_data)
# Plotting averages (using one of the metrics for demonstration)
plot_function(alpha_avg_data['alpha'], alpha_avg_data, "Volatility of Volatility", "Average Value", "Avg over Volatility", "alpha_avg")
plot_function(sigma_avg_data['sigma'], sigma_avg_data, "Volatility", "Average Value", "Avg over Trader Noise", "sigma_avg")

# # Preparing data for 3D plot (for 'loss_rl' as an example)
# loss_data = {alpha: [] for alpha in np.unique(data['alpha'])}
# for alpha in loss_data:
#     indices = np.where(data['alpha'] == alpha)
#     for sigma in np.unique(data['sigma']):
#         sigma_indices = np.where(data['sigma'][sigma_indices] == sigma)
#         intersection_indices = np.intersect1d(indices, sigma_indices)
#         if intersection_indices.size > 0:
#             loss_data[alpha].append(np.mean(np.array(data['loss_rl'])[intersection_indices]))
#         else:
#             loss_data[alpha].append(np.nan)  # Handling missing combinations

# # Extracting x, y, z for 3D plot
# alphas = list(loss_data.keys())
# sigmas = list(np.unique(data['sigma']))
# z_values = [loss_data[alpha] for alpha in alphas]

# # Creating 3D plot
# plot_3d(sigmas, alphas, z_values, "Sigma", "Alpha", "Loss", "3D Plot of Loss across Sigma and Alpha")

