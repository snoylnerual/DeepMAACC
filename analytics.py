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
import warnings
warnings.filterwarnings("ignore")



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu

vanilla = pd.read_csv('outputs/data/vanilla_experiments_obo.csv')
vl = len(vanilla.index)
approach1 = pd.read_csv('outputs/data/experiments_approach1_obo.csv')
a1l = len(approach1.index)
approach2 = pd.read_csv('outputs/data/experiments_approach2_obo.csv')
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
a1_params = pd.unique(approach1['Neurons_per_Cluster_param'])
a1_models = pd.unique(approach1['Model_Type'])
for model in a1_models:
	temp = []
	for n in a1_params:
		temp += [tuple((n,
				   [approach1[approach1['Neurons_per_Cluster_param'] == n][approach1['Model_Type'] == model]['Mutate_time'].mean(),
					approach1[approach1['Neurons_per_Cluster_param'] == n][approach1['Model_Type'] == model]['Cluster_time'].mean(),
					approach1[approach1['Neurons_per_Cluster_param'] == n][approach1['Model_Type'] == model]['Mutation_Score'].mean(),
					approach1[approach1['Neurons_per_Cluster_param'] == n][approach1['Model_Type'] == model]['MS_time'].mean()]))]
	avg_a1[model] = temp
	# Clusters_per_layer should be named Cluster_Sz
# [(2, fcnn-mnist, []), (2, fcnn-fmnist, [])]

avg_a2 = {}
a2_params = pd.unique(approach2['ParHAC_Threshold'])
a2_models = pd.unique(approach2['Model_Type'])
for model in a2_models:
	temp = []
	for n in a2_params:
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

#----------------------------------------------------------------------------------------------
#Approach2 Time Speedup

label = []
X = []
Y = []
for key, value in avg_a2.items():
	if key.split("/")[-1] in ['resnet18-cifar10.keras', 'resnet18-cifar100.keras', 'resnet18-svhn.keras']:
		continue
	for thing in value:
		label += [key]
		X += [thing[0]]
		Y += [float((avg_vanilla[key][2] - thing[1][3]) / avg_vanilla[key][2])]
		#Y += [float(((avg_vanilla[key][2]) - (thing[1][3]) / (avg_vanilla[key][2])))]
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
plt.xticks(np.array(a2_params))
plt.xlabel('PARHAC_Threshold')
plt.ylabel('Speedup')
plt.title('Approach 2: How much speedup you gain when using neuron clustering vs vanilla mutation testing?')
plt.legend(title='Models')
plt.grid(True)
plt.tight_layout()

# Show plot
plt.savefig('outputs/Approach2_TotalTimeSpeedup-vs-PARHAC.jpeg')

#----------------------------------------------------------------------------------------------
#Approach1 Time Speedup

label = []
X = []
Y = []
for key, value in avg_a1.items():
	if key.split("/")[-1] in ['resnet18-cifar10.keras', 'resnet18-cifar100.keras', 'resnet18-svhn.keras']:
		continue
	for thing in value:
		label += [key]
		X += [thing[0]]
		Y += [float((avg_vanilla[key][2] - thing[1][3]) / avg_vanilla[key][2])]

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
plt.xticks(np.array(a1_params))
plt.xlabel('Cluster Size')
plt.ylabel('Speedup')
plt.title('Approach 1: How much speedup you gain when using neuron clustering vs vanilla mutation testing?')
plt.legend(title='Models')
plt.grid(True)
plt.tight_layout()

plt.savefig('outputs/Approach1_TotalTimeSpeedup-vs-Clustersz.jpeg')

#===================================================================================================================================
#===================================================================================================================================
#===================================================================================================================================
#===================================================================================================================================
# RQ2

#----------------------------------------------------------------------------------------------
#Approach2 Mutation Score

label = []
X = []
Y = []
for key, value in avg_a2.items():
	if key.split("/")[-1] in ['resnet18-cifar10.keras', 'resnet18-cifar100.keras', 'resnet18-svhn.keras']:
		continue
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
plt.xticks(np.array(a2_params))
plt.xlabel('PARHAC_Threshold')
plt.ylabel('Mutation Score Error')
plt.title('Approach 2: How much mutation score is lost when using neuron clustering vs vanilla mutation testing?')
plt.legend(title='Models')
plt.grid(True)
plt.tight_layout()

# Show plot
plt.savefig('outputs/Approach2_MutScoreLoss-vs-PARHAC.jpeg')
#----------------------------------------------------------------------------------------------
#Approach1 Mutation Score

label = []
X = []
Y = []
for key, value in avg_a1.items():
	if key.split("/")[-1] in ['resnet18-cifar10.keras', 'resnet18-cifar100.keras', 'resnet18-svhn.keras']:
		continue
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
plt.xticks(np.array(a1_params))
plt.xlabel('Cluster Size')
plt.ylabel('Mutation Score Error')
plt.title('Approach 1: How much mutation score is lost when using neuron clustering vs vanilla mutation testing?')
plt.legend(title='Models')
plt.grid(True)
plt.tight_layout()
plt.savefig('outputs/Approach1_MutScoreLoss-vs-Clustersz.jpeg')

#=======================================================================
#----------------------------------------------------------------------------------------------
# Box and whisker plot
# one plot for each model and approach
# Approach 1 (fcnn-mnist, fcnn-fmnist, fcnn-kmnist, fcnn-emnist, lenet5-mnist, lenet5-fmnist, lenet5kmnist, lenet5-emnist, resnet18-cifar10, resnet18-svhn)
# Approach 2 (fcnn-mnist, fcnn-fmnist, fcnn-kmnist, fcnn-emnist, lenet5-mnist, lenet5-fmnist, lenet5kmnist, lenet5-emnist, resnet18-cifar10, resnet18-svhn)

#v_models = pd.unique(vanilla['Model_Type']) #approach1 & approach2
a1_params = pd.unique(approach1['Neurons_per_Cluster_param'])
#label = []
# X = []
# Y = []
#for key, value in avg_a1.items():
for model in a1_models:
	plt.figure(figsize=(10, 6))
	Y = []
	if model.split("/")[-1] in ['resnet18-cifar10.keras', 'resnet18-cifar100.keras', 'resnet18-svhn.keras']:
		continue
	for param in a1_params:
		a1_vals = approach1[approach1['Neurons_per_Cluster_param'] == param][approach1['Model_Type'] == model]['Mutate_time']
		temp = []
		for index, thing in a1_vals.items():
			temp += [float(((avg_vanilla[model][1]) - thing) / (avg_vanilla[model][1]))]
		Y += [temp]

	plt.boxplot(Y,labels = a1_params)

	plt.xlabel('Parameters')
	plt.ylabel('Speedup')
	plt.title(model.split("/")[-1]+' Box-and-Whisker Plot')
	plt.grid(True)
	plt.tight_layout()
	#
	# # Show plot
	plt.savefig('outputs/Approach1_'+model.split("/")[-1]+'_BoxPlot-Y-Values.jpeg')

a2_params = pd.unique(approach2['ParHAC_Threshold'])
for model in a2_models:
	plt.figure(figsize=(10, 6))
	Y = []
	if model.split("/")[-1] in ['resnet18-cifar10.keras', 'resnet18-cifar100.keras', 'resnet18-svhn.keras']:
		continue
	for param in a2_params:
		a2_vals = approach2[approach2['ParHAC_Threshold'] == param][approach2['Model_Type'] == model]['Mutate_time']
		temp = []
		for index, thing in a2_vals.items():
			temp += [float(((avg_vanilla[model][1]) - thing) / (avg_vanilla[model][1]))]
		Y += [temp]

	plt.boxplot(Y,labels = a2_params)

	plt.xlabel('Parameters')
	plt.ylabel('Speedup')
	plt.title(model.split("/")[-1]+' Box-and-Whisker Plot')
	plt.grid(True)
	plt.tight_layout()
	#
	# # Show plot
	plt.savefig('outputs/Approach2_'+model.split("/")[-1]+'_BoxPlot-Y-Values.jpeg')


#=======================================================================
#----------------------------------------------------------------------------------------------
# Mann-Whitney U-test Approach1/2 for both Mutation Test and Time and in between parameters for each model
# # U1, p = mannwhitneyu()
# P-value significantly smaller than 0.05 will indicate/signify statistical significance with a confidence level of 95%

X = np.array(X)
Y = np.array(Y)
labels = np.array(label)
mwut_df = pd.DataFrame(columns=['X_Model', 'Y_Model', 'x_parameter', 'y_parameter', 'U1', 'p',])

for modelx in v_models:
	for modely in v_models:
		for paramy in a1_params:
			x = vanilla[vanilla['Model_Type'] == modelx]['Mutate_time']
			y = approach1[approach1['Neurons_per_Cluster_param'] == paramy][approach1['Model_Type'] == modely]['Mutate_time']
			U1,p = mannwhitneyu(x, y)
			mwut_df.loc[len(mwut_df.index)] = [modelx.split("/")[-1], modely.split("/")[-1], 0, paramy, U1, p]
		for paramy in a2_params:
			x = vanilla[vanilla['Model_Type'] == modelx]['Mutate_time']
			y = approach2[approach2['ParHAC_Threshold'] == paramy][approach2['Model_Type'] == modely]['Mutate_time']
			U1,p = mannwhitneyu(x, y)
			mwut_df.loc[len(mwut_df.index)] = [modelx.split("/")[-1], modely.split("/")[-1], 0, paramy, U1, p]
	for index, paramx in enumerate(a1_params):
		for i in range(index+1, len(a1_params)):
			x = approach1[approach1['Neurons_per_Cluster_param'] == paramx][approach1['Model_Type'] == modelx]['Mutate_time']
			y = approach1[approach1['Neurons_per_Cluster_param'] == a1_params[i]][approach1['Model_Type'] == modelx]['Mutate_time']
			U1,p = mannwhitneyu(x, y)
			mwut_df.loc[len(mwut_df.index)] = [modelx.split("/")[-1], modelx.split("/")[-1], paramx, a1_params[i], U1, p]
	for index, paramx in enumerate(a2_params):
		for i in range(index+1, len(a2_params)):
			x = approach2[approach2['ParHAC_Threshold'] == paramx][approach2['Model_Type'] == modelx]['Mutate_time']
			y = approach2[approach2['ParHAC_Threshold'] == a2_params[i]][approach2['Model_Type'] == modelx]['Mutate_time']
			U1,p = mannwhitneyu(x, y)
			mwut_df.loc[len(mwut_df.index)] = [modelx.split("/")[-1], modelx.split("/")[-1], paramx, a2_params[i], U1, p]
	for index, paramx in enumerate(a1_params):
		for i in range(index+1, len(a2_params)):
			x = approach1[approach1['Neurons_per_Cluster_param'] == paramx][approach1['Model_Type'] == modelx]['Mutate_time']
			y = approach2[approach2['ParHAC_Threshold'] == a2_params[i]][approach2['Model_Type'] == modelx]['Mutate_time']
			U1,p = mannwhitneyu(x, y)
			mwut_df.loc[len(mwut_df.index)] = [modelx.split("/")[-1], modelx.split("/")[-1], paramx, a2_params[i], U1, p]

mwut_df.to_csv('outputs/MannWhitneyUTestValues.csv', mode='w', header=True, index=False)

# fig, ax = plt.subplots()
#
# #hide the axes
# fig.patch.set_visible(False)
# ax.axis('off')
# ax.axis('tight')
# table = ax.table(cellText=mwut_df.values, colLabels=mwut_df.columns, loc='center')
#
# #display table
# fig.tight_layout()
# plt.savefig('outputs/MannWhitneyUTestValues.jpeg')
