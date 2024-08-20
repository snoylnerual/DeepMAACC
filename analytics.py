## How much speedup you gain when using neuron clustering vs vanilla mutation testing?
# 	- Mutation testing time for vanilla, approach1, approach2
# 	- ((vanilla mutation testing time) - (approach-1-2-mutation testing time)) / (vanilla mutation testing time)
# 	- => gives the speedup for approach1 or approach2
# 	- Percentage of improvement in time
# 	- Use the average of the multiple runs(6 runs) to plot it
# 	- Store the raw data into a csv file
# 	- Plot of colored vanilla&approach1, vanilla&approach2
# 	- Y is percentage of improvement
# 	- X is param (cluster_sz, parhac_threshold)
#
# How much mutation score is lost when using neuron clustering vs vanilla mutation testing?
# 	- ((vanilla mutation score) - (approach-1-2-mutation score)) / (vanilla mutation score)
# 	- Percentage of deviation of the mutation score for each approach (from vanilla approach)
# 	- Similar plots as RQ1
#
#
# What is the impact of non-determinacy in training process on mutation score of your tool when using neuron clustering vs in vanilla mode
# 	- No more non-determinacy from GF mutator
# 	- The averaging of the 6 runs should remove non-determinacy
# 	- Whether the improvement we see is it of statistical significance: is this outcome by chance or the output
#
#
# Mann-Whitney U-test using scipy
# -> it will return p_value from csv files
# For each param value for approach1/2, you need a p-value (record that)
# (P-value significantly smaller than 0.05 will indicate/signify statistical significance with a confidence level of 95%
# (example outcome sentence: we observe that all outputs show a p-value of less than 0.05)
#




import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu

vanilla = pd.read_csv('outputs/vanilla_experiments_obo.csv')
vl = len(vanilla.index)
approach1 = pd.read_csv('outputs/experiments_approach1_obo.csv')
a1l = len(approach1.index)
approach2 = pd.read_csv('outputs/experiments_approach2_obo.csv')
a2l = len(approach2.index)

# for i in (v1):
# 	vanilla.iloc[i][['','']]



avg_vanilla = {}
v_models = pd.unique(vanilla['Model_Type'])
for model in v_models:
	avg_vanilla[model] = [vanilla[vanilla['Model_Type'] == model]['Mutate_time'].mean(),
						  vanilla[vanilla['Model_Type'] == model]['Mutation_Score'].mean(),
						  vanilla[vanilla['Model_Type'] == model]['MS_time'].mean()]


avg_a1 = {}
a1_models = pd.unique(approach1['Model_Type'])
for model in a1_models:
	temp = []
	for n in [2, 4, 6, 8, 10]:
		temp += [tuple((n,
				   [approach1[approach1['Neurons_per_Cluster_param'] == n][approach1['Model_Type'] == model]['Mutate_time'].mean(),
					approach1[approach1['Neurons_per_Cluster_param'] == n][approach1['Model_Type'] == model]['Cluster_time'].mean(),
					approach1[approach1['Neurons_per_Cluster_param'] == n][approach1['Model_Type'] == model]['Mutation_Score'].mean(),
					approach1[approach1['Neurons_per_Cluster_param'] == n][approach1['Model_Type'] == model]['MS_time'].mean()]))]
	avg_a1[model] = temp
	# Clusters_per_layer should be named Cluster_Sz
# [(2, fcnn-mnist, []), (2, fcnn-fmnist, [])]

avg_a2 = {}
a2_models = pd.unique(approach2['Model_Type'])
for model in a2_models:
	temp = []
	for n in [3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7]:
		temp += [tuple((n,
				   [approach2[approach2['ParHAC_Threshold'] == n][approach2['Model_Type'] == model]['Mutate_time'].mean(),
				   approach2[approach2['ParHAC_Threshold'] == n][approach2['Model_Type'] == model]['Cluster_time'].mean(),
				   approach2[approach2['ParHAC_Threshold'] == n][approach2['Model_Type'] == model]['Mutation_Score'].mean(),
				   approach2[approach2['ParHAC_Threshold'] == n][approach2['Model_Type'] == model]['MS_time'].mean()]))]
	avg_a2[model] = temp

# [(3, fcnn-mnist, []), (3, fcnn-fmnist, [])]

#for key, value in avg_a1.items():

#===================================================================================================================================
#===================================================================================================================================
#===================================================================================================================================
#===================================================================================================================================
# RQ1

label = []
X = []
Y = []
for key, value in avg_a2.items():
	for thing in value:
		label += [key]
		X += [thing[0]]
		Y += [float(((avg_vanilla[key][2]) - (thing[1][3]) / (avg_vanilla[key][2])))]
		#Y += [float(((avg_vanilla[key][0] + avg_vanilla[key][2]) - (thing[1][0] + thing[1][1] + thing[1][3])) / (avg_vanilla[key][0] + avg_vanilla[key][2]))]
		#((vanilla mutation testing time) - (approach-1-2-mutation testing time)) / (vanilla mutation testing time)


# Convert lists to numpy arrays for easier manipulation
X = np.array(X)
Y = np.array(Y)
labels = np.array(label)

# Remove NaN values
valid_indices = ~np.isnan(Y)
X_clean = X[valid_indices]
Y_clean = Y[valid_indices]
labels_clean = labels[valid_indices]

# Plotting
plt.figure(figsize=(10, 6))

# Get unique labels
unique_labels = np.unique(labels_clean)

# Plot each label group separately
for label in unique_labels:
    indices = labels_clean == label
    plt.plot(X_clean[indices], Y_clean[indices], label=label)

# Customizing the plot
plt.xlabel('PARHAC_Threshold')
plt.ylabel('Speedup')
plt.title('Approach 2: How much speedup you gain when using neuron clustering vs vanilla mutation testing?')
plt.legend(title='Models')
plt.grid(True)
plt.tight_layout()

# Show plot
plt.savefig('Approach2_TotalTimeSpeedup-vs-PARHAC.jpeg')

label = []
X = []
Y = []
for key, value in avg_a1.items():
	for thing in value:
		label += [key]
		X += [thing[0]]
		Y += [float(((avg_vanilla[key][2]) - (thing[1][3]) / (avg_vanilla[key][2])))]

X = np.array(X)
Y = np.array(Y)
labels = np.array(label)

# Remove NaN values
valid_indices = ~np.isnan(Y)
X_clean = X[valid_indices]
Y_clean = Y[valid_indices]
labels_clean = labels[valid_indices]

# Plotting
plt.figure(figsize=(10, 6))

# Get unique labels
unique_labels = np.unique(labels_clean)

# Plot each label group separately
for label in unique_labels:
    indices = labels_clean == label
    plt.plot(X_clean[indices], Y_clean[indices], label=label)

# Customizing the plot
plt.xlabel('Cluster Size')
plt.ylabel('Speedup')
plt.title('Approach 1: How much speedup you gain when using neuron clustering vs vanilla mutation testing?')
plt.legend(title='Models')
plt.grid(True)
plt.tight_layout()

plt.savefig('Approach1_TotalTimeSpeedup-vs-Clustersz.jpeg')

#===================================================================================================================================
#===================================================================================================================================
#===================================================================================================================================
#===================================================================================================================================
# RQ2

label = []
X = []
Y = []
for key, value in avg_a2.items():
	for thing in value:
		label += [key]
		X += [thing[0]]
		#Y += [float(((avg_vanilla[key][2]) - (thing[1][3]) / (avg_vanilla[key][2])))]
		Y += [float(((avg_vanilla[key][1]) - (thing[1][2])) / (avg_vanilla[key][1]))]
# ((vanilla mutation testing time) - (approach-1-2-mutation testing time)) / (vanilla mutation testing time)

# Convert lists to numpy arrays for easier manipulation
X = np.array(X)
Y = np.array(Y)
labels = np.array(label)

# Remove NaN values
valid_indices = ~np.isnan(Y)
X_clean = X[valid_indices]
Y_clean = Y[valid_indices]
labels_clean = labels[valid_indices]

# Plotting
plt.figure(figsize=(10, 6))

# Get unique labels
unique_labels = np.unique(labels_clean)

# Plot each label group separately
for label in unique_labels:
	indices = labels_clean == label
	plt.plot(X_clean[indices], Y_clean[indices], label=label)

# Customizing the plot
plt.xlabel('PARHAC_Threshold')
plt.ylabel('Mutation Score Error')
plt.title('Approach 2: How much mutation score is lost when using neuron clustering vs vanilla mutation testing?')
plt.legend(title='Models')
plt.grid(True)
plt.tight_layout()

# Show plot
plt.savefig('Approach2_MutScoreLoss-vs-PARHAC.jpeg')

label = []
X = []
Y = []
for key, value in avg_a1.items():
	for thing in value:
		label += [key]
		X += [thing[0]]
		Y += [float(((avg_vanilla[key][1]) - (thing[1][2])) / (avg_vanilla[key][1]))]

X = np.array(X)
Y = np.array(Y)
labels = np.array(label)

# Remove NaN values
valid_indices = ~np.isnan(Y)
X_clean = X[valid_indices]
Y_clean = Y[valid_indices]
labels_clean = labels[valid_indices]

# Plotting
plt.figure(figsize=(10, 6))

# Get unique labels
unique_labels = np.unique(labels_clean)

# Plot each label group separately
for label in unique_labels:
	indices = labels_clean == label
	plt.plot(X_clean[indices], Y_clean[indices], label=label)

# Customizing the plot
plt.xlabel('Cluster Size')
plt.ylabel('Speedup')
plt.title('Approach 1: How much mutation score is lost when using neuron clustering vs vanilla mutation testing?')
plt.legend(title='Models')
plt.grid(True)
plt.tight_layout()
plt.savefig('Approach1_MutScoreLoss-vs-Clustersz.jpeg')

#=======================================================================


#label = []
X = []
Y = []
for key, value in avg_a1.items():
	for thing in value:
		#label += [key]
		X += [thing[0]]
		Y += [float(((avg_vanilla[key][1]) - (thing[1][2])) / (avg_vanilla[key][1]))]

# Convert lists to numpy arrays for easier manipulation
Y = np.array(Y)
X = np.array(X)
labels = np.array(labels)

# # Remove NaN values
valid_indices = ~np.isnan(Y)
Y_clean = Y[valid_indices]
labels_clean = labels[valid_indices]

# Prepare data for box plot
unique_labels = np.unique(labels_clean)
data_for_boxplot = [Y_clean[labels_clean == label] for X in unique_labels]

# Plotting
plt.figure(figsize=(10, 6))
plt.boxplot(data_for_boxplot, labels=unique_labels, vert=True)

# Customizing the plot
plt.xlabel('Parameters')
plt.ylabel('Speedup')
plt.title('Box-and-Whisker Plot')
plt.grid(True)
plt.tight_layout()
#
# # Show plot
plt.savefig('Approach2_BoxPlot-Y-Values.jpeg')
#
#
# label = []
# X = []
# Y = []
# approach1['Mutate_time']
# approach2['Mutate_time']
# for key, value in avg_a2.items():
# 	for thing in value:
# 		label += [key]
# 		X += [thing[0]]
# 		Y += [float(((avg_vanilla[key][1]) - (thing[1][2])) / (avg_vanilla[key][1]))]
#
#
# X = np.array()
# Y = np.array(Y)
#
# # Remove NaN values from Y
# valid_indices = ~np.isnan(Y)
# X_clean = X[valid_indices]
# Y_clean = Y[valid_indices]
#
# # Prepare data for box plot
# data_for_boxplot = [X_clean, Y_clean]
#
# # Plotting
# plt.figure(figsize=(8, 6))
# plt.boxplot(data_for_boxplot, labels=['X Values', 'Y Values'], vert=True)
#
# # Customizing the plot
# plt.ylabel('Values')
# plt.title('Box-and-Whisker Plot for X and Y')
# plt.grid(True)
# plt.tight_layout()
#
# # Show plot
# plt.show()
#
#
# # U1, p = mannwhitneyu()
# #
# # for t in avg_a1:
#
#
#
