#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# custom functions #
import utils


# In[2]:


# If running this interactively, can specify jupyter_args here for argparser to use
if utils.is_interactive():
    methods = "pretrained_subj01_40sess_hypatia_vd2,method2,method3"
    data_path = "/weka/proj-medarc/shared/mindeyev2_dataset"
    criteria = "all"
    print("Methods:", methods)

    jupyter_args = f"--methods={methods} --data_path={data_path} --criteria={criteria}"
    print(jupyter_args)
    jupyter_args = jupyter_args.split()
    
    from IPython.display import clear_output  # Function to clear print outputs in cell
    get_ipython().run_line_magic('load_ext', 'autoreload')
    # This allows you to change functions in models.py or utils.py and have this notebook automatically update with your revisions
    get_ipython().run_line_magic('autoreload', '2')


# In[3]:


parser = argparse.ArgumentParser(description="Compare methods based on metrics")
parser.add_argument(
    "--methods", type=str, required=True,
    help="Comma-separated list of method names to compare",
)
parser.add_argument(
    "--data_path", type=str, default="../dataset",
    help="Path to where metrics CSV files are stored",
)
parser.add_argument(
    "--columns_to_normalize", type=str, default='PixCorr,SSIM,AlexNet(2),AlexNet(5),InceptionV3,CLIP,EffNet-B,SwAV,FwdRetrieval,BwdRetrieval,Brain Corr. nsd_general,Brain Corr. V1,Brain Corr. V2,Brain Corr. V3,Brain Corr. V4,Brain Corr. higher_vis',
    help="Comma-separated list of metric columns to normalize",
)
parser.add_argument(
    "--criteria", type=str, default="all",
    help="Criteria to use for averaging metrics. 'all' or comma-separated list of metrics",
)
parser.add_argument(
    "--output_path", type=str, default="../figs",
    help="Path to save the output scatter plot",
)
parser.add_argument(
    "--output_file", type=str, default="method_scatter_plot",
    help="Filename to save the output scatter plot",
)
        
if utils.is_interactive():
    args = parser.parse_args(jupyter_args)
else:
    args = parser.parse_args()

# Create global variables without the args prefix
for attribute_name in vars(args).keys():
    globals()[attribute_name] = getattr(args, attribute_name)

criteria = criteria.replace("*", " ")
methods = [method.strip() for method in methods.split(",")]
columns_to_normalize = [col.strip() for col in columns_to_normalize.split(",")]

# Seed all random functions
seed = 42  # Set your seed value
utils.seed_everything(seed)


# # Loading tables

# In[4]:


# Loading tables for both modes
modes = ['imagery', 'vision']
dfs = []
for method in methods:
    for mode in modes:
        metrics_file = f"tables/{method}_all_recons_{mode}.csv"
        if not os.path.exists(metrics_file):
            print(f"Metrics file for method '{method}' and mode '{mode}' not found at {metrics_file}")
            sys.exit(1)
        df = pd.read_csv(metrics_file, sep="\t")
        df['method'] = method
        df['mode'] = mode
        dfs.append(df)

all_metrics = pd.concat(dfs, ignore_index=True)


# In[6]:


# Check that columns_to_normalize exist in DataFrame
missing_columns = [col for col in columns_to_normalize if col not in all_metrics.columns]
if missing_columns:
    print(f"Error: The following columns to normalize are missing from the data: {missing_columns}")
    sys.exit(1)

# Normalize specified columns across the entire dataset
scaler = MinMaxScaler()
all_metrics[columns_to_normalize] = scaler.fit_transform(all_metrics[columns_to_normalize])

# Determine metrics to average
if criteria == 'all':
    metrics_to_average = columns_to_normalize
else:
    metrics_to_average = [col.strip() for col in criteria.split(",")]

# Check that metrics_to_average exist in DataFrame
missing_columns = [col for col in metrics_to_average if col not in all_metrics.columns]
if missing_columns:
    print(f"Error: The following metrics are missing from the data: {missing_columns}")
    sys.exit(1)

# Compute average normalized metric performance per method and mode
method_mode_scores = all_metrics.groupby(['method', 'mode'])[metrics_to_average].mean()
method_mode_scores['average_score'] = method_mode_scores.mean(axis=1)

# In[6]:

# Create a pivot table with methods as index and modes as columns
average_scores = method_mode_scores['average_score'].unstack()

# Ensure that both 'imagery' and 'vision' modes are present for all methods
average_scores = average_scores.dropna()


# In[12]:


# Plot scatter plot
plt.figure(figsize=(10, 6))
ax = plt.subplot(111)
colors = plt.cm.tab10.colors  # Get 10 colors from colormap
for i, method in enumerate(average_scores.index):
    x = average_scores.loc[method, 'vision']
    y = average_scores.loc[method, 'imagery']
    plt.scatter(x, y, color=colors[i % len(colors)], label=method, s=100)

plt.xlabel('Vision Performance')
plt.ylabel('Imagery Performance')
plt.title(f'Imagery vs. Vision Performance\n{output_file}')
box = ax.get_position()
ax.set_position([box.x0 - 0.06, box.y0, box.width * 0.6, box.height])
# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True)
# plt.tight_layout()
output_file = os.path.join(output_path, f'{output_file}.png')
print(f"Saving scatter plot to {output_file}")
plt.savefig(output_file, dpi=300)
plt.show()


# ### 
