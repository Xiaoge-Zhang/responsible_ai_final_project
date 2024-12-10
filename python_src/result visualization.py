import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Metrics and groups
models = ['SVM', 'Random_Forest', 'Naive_Bayes', 'Logistic_Regression', 'Decision_Tree']
protected_attrs = ['race']
metrics = ["dir_sex", "dir_race", "aoe_sex", "aoe_race", "aoe_combined"]
output_dir = '../output/'
visualization_dir = '../visualization/'

groups = [
    "Original Performance",
    "Performance After Feature Reweighting",
    "Performance After Exponentiated Gradient Reduction",
    "Performance After Grid Search",
    "Improvement Feature Reweighting",
    "Improvement Exponentiated Gradient Reduction",
    "Improvement Grid Search",
]

data = {}
data['Original Performance'] = []
data['Performance After Feature Reweighting'] = []
data['Performance After Exponentiated Gradient Reduction'] = []
data['Performance After Grid Search'] = []

for idx, model in enumerate(models):
    file_dir = output_dir + model + '_' + '_'.join(protected_attrs) + '.csv'
    df = pd.read_csv(file_dir)
    data['Original Performance'].append(df['default'])
    data['Performance After Feature Reweighting'].append(df['reweight'])
    data['Performance After Exponentiated Gradient Reduction'].append(df['egr'])
    data['Performance After Grid Search'].append(df['gsr'])

data['Original Performance'] = np.transpose(data['Original Performance']).tolist()
data['Performance After Feature Reweighting'] = np.transpose(data['Performance After Feature Reweighting']).tolist()
data['Performance After Exponentiated Gradient Reduction'] = np.transpose(data['Performance After Exponentiated Gradient Reduction']).tolist()
data['Performance After Grid Search'] = np.transpose(data['Performance After Grid Search']).tolist()

# Compute improvements dynamically
data["Improvement Feature Reweighting"] = [
    np.array(original) - np.array(after_A)
    for original, after_A in zip(data["Original Performance"], data["Performance After Feature Reweighting"])
]
data["Improvement Exponentiated Gradient Reduction"] = [
    np.array(original) - np.array(after_B)
    for original, after_B in zip(data["Original Performance"], data["Performance After Exponentiated Gradient Reduction"])
]
data["Improvement Grid Search"] = [
    np.array(original) - np.array(after_C)
    for original, after_C in zip(data["Original Performance"], data["Performance After Grid Search"])
]

# Calculate averages for each metric and group
averages = {group: [np.mean(values) for values in metrics_data] for group, metrics_data in data.items()}

# Prepare data for plotting
x = np.arange(len(metrics))  # Positions for metrics
width = 0.12  # Width of each bar

# Create the figure and axes
fig, ax = plt.subplots(figsize=(12, 6))

# Plot each group's data as a bar
for i, group in enumerate(groups):
    ax.bar(
        x + i * width,
        averages[group],
        width,
        label=group,
    )

# Add labels, title, and legend
ax.set_xlabel("Metrics", fontsize=12)
ax.set_ylabel("Average Values", fontsize=12)
ax.set_title("Average Performance and Improvement Across Metrics", fontsize=14)
ax.set_xticks(x + width * (len(groups) - 1) / 2)  # Center the tick labels
ax.set_xticklabels(metrics, fontsize=10)
ax.legend(title="Groups", fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.7)

# Show the plot
plt.tight_layout()
plt.savefig(visualization_dir + '_'.join(protected_attrs) + '.pdf', format='pdf')
plt.show()
