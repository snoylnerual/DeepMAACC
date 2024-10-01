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
from matplotlib import rcParams

rcParams['font.weight'] = 'bold'

plt.rc('font',family='Times New Roman')

vanilla = pd.read_csv('outputs/data/final_vanilla_experiments.csv')
vl = len(vanilla.index)
approach1 = pd.read_csv('outputs/data/final_approach1_experiments.csv')
a1l = len(approach1.index)
approach2 = pd.read_csv('outputs/data/final_approach2_experiments.csv')
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
a1_params.sort()
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
#a2_params = pd.unique(approach2['ParHAC_Threshold'])
a2_params = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
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

print("Average speedup of Approach2 across all fcnn&lenet5 models and all parameters is: " + str(Y.mean()))

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

print("Average speedup of Approach1 across all fcnn&lenet5 models and all parameters is: " + str(Y.mean()))


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

print("Average MSE of Approach2 across all fcnn&lenet5 models and all parameters is: " + str(Y.mean()))

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

print("Average MSE of Approach1 across all fcnn&lenet5 models and all parameters is: " + str(Y.mean()))


#----------------------------------------------------------------------------------------------
#Approach2 Time Speedup

label = []
X = []
Y = []
for key, value in avg_a2.items():
	# if key.split("/")[-1] in ['resnet18-cifar10.keras', 'resnet18-cifar100.keras', 'resnet18-svhn.keras']:
	# 	continue
	if 'fcnn' in key:
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

print("Average speedup of Approach2 across all fcnn models and all parameters is: " + str(Y.mean()))

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
    plt.plot(X_clean[indices], Y_clean[indices], label=label.split('/')[-1].split('.')[0].upper())

# Customizing the plot
plt.xticks(np.array(a2_params))
plt.xlabel('PARHAC Threshold', fontsize=24, weight='bold')
plt.ylabel('Speedup', fontsize=24, weight='bold')
# plt.title('Approach 2: How much speedup you gain when using neuron clustering vs vanilla mutation testing?')
plt.legend(title='Models', fontsize=22)
plt.grid(True)
plt.tight_layout()

# Show plot
plt.savefig('outputs/Approach2_TotalTimeSpeedup-vs-PARHAC_FCNN.pdf')



label = []
X = []
Y = []
for key, value in avg_a2.items():
	# if key.split("/")[-1] in ['resnet18-cifar10.keras', 'resnet18-cifar100.keras', 'resnet18-svhn.keras']:
	# 	continue
	if 'lenet' in key:
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

print("Average speedup of Approach2 across all lenet5 models and all parameters is: " + str(Y.mean()))

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
    plt.plot(X_clean[indices], Y_clean[indices], label=label.split('/')[-1].split('.')[0].upper().replace('LENET', 'LeNet-'))

# Customizing the plot
plt.xticks(np.array(a2_params))
plt.xlabel('PARHAC Threshold', fontsize=24, weight='bold')
plt.ylabel('Speedup', fontsize=24, weight='bold')
# plt.title('Approach 2: How much speedup you gain when using neuron clustering vs vanilla mutation testing?')
plt.legend(title='Models', fontsize=22)
plt.grid(True)
plt.tight_layout()

# Show plot
plt.savefig('outputs/Approach2_TotalTimeSpeedup-vs-PARHAC_LeNet5.pdf')

#----------------------------------------------------------------------------------------------
#Approach1 Time Speedup

label = []
X = []
Y = []
for key, value in avg_a1.items():
	# if key.split("/")[-1] in ['resnet18-cifar10.keras', 'resnet18-cifar100.keras', 'resnet18-svhn.keras']:
	# 	continue
	if 'fcnn' in key:
		for thing in value:
			label += [key]
			X += [thing[0]]
			Y += [float((avg_vanilla[key][2] - thing[1][3]) / avg_vanilla[key][2])]


X = np.array(X)
Y = np.array(Y)
labels = np.array(label)

print("Average speedup of Approach1 across all fcnn models and all parameters is: " + str(Y.mean()))

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
	plt.plot(X_clean[indices], Y_clean[indices], label=label.split('/')[-1].split('.')[0].upper())
#
# print(X)
#
# plt.plot(X, Y, label=label)

# Customizing the plot
plt.xticks(np.array(a1_params))
plt.xlabel('Neurons per Cluster', fontsize=24, weight='bold')
plt.ylabel('Speedup', fontsize=24, weight='bold')
# plt.title('Approach 1: How much speedup you gain when using neuron clustering vs vanilla mutation testing?')
plt.legend(title='Models', fontsize=22)
plt.grid(True)
plt.tight_layout()

plt.savefig('outputs/Approach1_TotalTimeSpeedup-vs-Clustersz_FCNN.pdf')



label = []
X = []
Y = []
for key, value in avg_a1.items():
	# if key.split("/")[-1] in ['resnet18-cifar10.keras', 'resnet18-cifar100.keras', 'resnet18-svhn.keras']:
	# 	continue
	if 'lenet' in key:
		for thing in value:
			label += [key]
			X += [thing[0]]
			Y += [float((avg_vanilla[key][2] - thing[1][3]) / avg_vanilla[key][2])]


X = np.array(X)
Y = np.array(Y)
labels = np.array(label)

print("Average speedup of Approach1 across all lenet5 models and all parameters is: " + str(Y.mean()))

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
	plt.plot(X_clean[indices], Y_clean[indices], label=label.split('/')[-1].split('.')[0].upper().replace('LENET', 'LeNet-'))
#
# print(X)
#
# plt.plot(X, Y, label=label)

# Customizing the plot
plt.xticks(np.array(a1_params))
plt.xlabel('Neurons per Cluster', fontsize=24, weight='bold')
plt.ylabel('Speedup', fontsize=24, weight='bold')
# plt.title('Approach 1: How much speedup you gain when using neuron clustering vs vanilla mutation testing?')
plt.legend(title='Models', fontsize=22)
plt.grid(True)
plt.tight_layout()

plt.savefig('outputs/Approach1_TotalTimeSpeedup-vs-Clustersz_LeNet5.pdf')

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
	# if key.split("/")[-1] in ['resnet18-cifar10.keras', 'resnet18-cifar100.keras', 'resnet18-svhn.keras']:
	# 	continue
	if 'fcnn' in key:
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

print("Average MSE of Approach2 across all fcnn models and all parameters is: " + str(Y.mean()))

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
	plt.plot(X_clean[indices], Y_clean[indices], label=label.split('/')[-1].split('.')[0].upper())

# Customizing the plot
plt.xticks(np.array(a2_params))
plt.xlabel('PARHAC Threshold', fontsize=24, weight='bold')
plt.ylabel('Mutation Score Error', fontsize=24, weight='bold')
# plt.title('Approach 2: How much mutation score is lost when using neuron clustering vs vanilla mutation testing?')
plt.legend(title='Models', fontsize=22)
plt.grid(True)
plt.tight_layout()

# Show plot
plt.savefig('outputs/Approach2_MutScoreLoss-vs-PARHAC_FCNN.pdf')

label = []
X = []
Y = []
for key, value in avg_a2.items():
	# if key.split("/")[-1] in ['resnet18-cifar10.keras', 'resnet18-cifar100.keras', 'resnet18-svhn.keras']:
	# 	continue
	if 'lenet5' in key:
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

print("Average MSE of Approach2 across all lenet5 models and all parameters is: " + str(Y.mean()))

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
	plt.plot(X_clean[indices], Y_clean[indices], label=label.split('/')[-1].split('.')[0].upper().replace('LENET', 'LeNet-'))

# Customizing the plot
plt.xticks(np.array(a2_params))
plt.xlabel('PARHAC Threshold', fontsize=24, weight='bold')
plt.ylabel('Mutation Score Error', fontsize=24, weight='bold')
# plt.title('Approach 2: How much mutation score is lost when using neuron clustering vs vanilla mutation testing?')
plt.legend(title='Models', fontsize=22)
plt.grid(True)
plt.tight_layout()

# Show plot
plt.savefig('outputs/Approach2_MutScoreLoss-vs-PARHAC_LeNet-5.pdf')
#----------------------------------------------------------------------------------------------
#Approach1 Mutation Score Error

label = []
X = []
Y = []
for key, value in avg_a1.items():
	# if key.split("/")[-1] in ['resnet18-cifar10.keras', 'resnet18-cifar100.keras', 'resnet18-svhn.keras']:
	# 	continue
	if 'fcnn' in key:
		for thing in value:
			label += [key]
			X += [thing[0]]
			Y += [float(((avg_vanilla[key][1]) - (thing[1][2])) / (avg_vanilla[key][1]))]

X = np.array(X)
Y = np.array(Y)
labels = np.array(label)

print("Average MSE of Approach1 across all fcnn models and all parameters is: " + str(Y.mean()))

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
	plt.plot(X_clean[indices], Y_clean[indices], label=label.split('/')[-1].split('.')[0].upper())

# Customizing the plot
plt.xticks(np.array(a1_params))
plt.xlabel('Neurons per Cluster', fontsize=24, weight='bold')
plt.ylabel('Mutation Score Error', fontsize=24, weight='bold')
# plt.title('Approach 1: How much mutation score is lost when using neuron clustering vs vanilla mutation testing?')
plt.legend(title='Models', fontsize=22)
plt.grid(True)
plt.tight_layout()
plt.savefig('outputs/Approach1_MutScoreLoss-vs-Clustersz_FCNN.pdf')

label = []
X = []
Y = []
for key, value in avg_a1.items():
	# if key.split("/")[-1] in ['resnet18-cifar10.keras', 'resnet18-cifar100.keras', 'resnet18-svhn.keras']:
	# 	continue
	if 'lenet' in key:
		for thing in value:
			label += [key]
			X += [thing[0]]
			Y += [float(((avg_vanilla[key][1]) - (thing[1][2])) / (avg_vanilla[key][1]))]

X = np.array(X)
Y = np.array(Y)
labels = np.array(label)

print("Average MSE of Approach1 across all lenet5 models and all parameters is: " + str(Y.mean()))

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
	plt.plot(X_clean[indices], Y_clean[indices], label=label.split('/')[-1].split('.')[0].upper().replace('LENET', 'LeNet-'))

# Customizing the plot
plt.xticks(np.array(a1_params))
plt.xlabel('Neurons per Cluster', fontsize=24, weight='bold')
plt.ylabel('Mutation Score Error', fontsize=24, weight='bold')
# plt.title('Approach 1: How much mutation score is lost when using neuron clustering vs vanilla mutation testing?')
plt.legend(title='Models', fontsize=22)
plt.grid(True)
plt.tight_layout()
plt.savefig('outputs/Approach1_MutScoreLoss-vs-Clustersz_LeNet5.pdf')

#----------------------------------------------------------------------------------------------
#Approach1 Mutation Score Increase/Decrease with cluster size plot

#=======================================================================
#----------------------------------------------------------------------------------------------
# Box and whisker plot
# one plot for each model and approach
# Approach 1 (fcnn-mnist, fcnn-fmnist, fcnn-kmnist, fcnn-emnist, lenet5-mnist, lenet5-fmnist, lenet5kmnist, lenet5-emnist, resnet18-cifar10, resnet18-svhn)
# Approach 2 (fcnn-mnist, fcnn-fmnist, fcnn-kmnist, fcnn-emnist, lenet5-mnist, lenet5-fmnist, lenet5kmnist, lenet5-emnist, resnet18-cifar10, resnet18-svhn)

#v_models = pd.unique(vanilla['Model_Type']) #approach1 & approach2
# a1_params = pd.unique(approach1['Neurons_per_Cluster_param'])
# a1_params.sort()
# #label = []
# # X = []
# # Y = []
# #for key, value in avg_a1.items():
# for model in a1_models:
# 	if model.split("/")[-1] in ['resnet18-cifar10.keras', 'resnet18-cifar100.keras', 'resnet18-svhn.keras']:
# 		continue
# 	plt.figure(figsize=(10, 6))
# 	Y = []
# 	for param in a1_params:
# 		a1_vals = approach1[approach1['Neurons_per_Cluster_param'] == param][approach1['Model_Type'] == model]['MS_time']
# 		temp = []
# 		for index, thing in a1_vals.items():
# 			temp += [float(((avg_vanilla[model][2]) - thing) / (avg_vanilla[model][2]))]
# 		Y += [temp]
#
# 	plt.boxplot(Y,labels = a1_params)
#
# 	plt.xlabel('Parameters', fontsize=20)
# 	plt.ylabel('Speedup')
# 	# plt.title(model.split("/")[-1]+' Box-and-Whisker Plot')
# 	plt.grid(True)
# 	plt.tight_layout()
# 	#
# 	# # Show plot
# 	plt.savefig('outputs/Approach1_'+model.split("/")[-1]+'_BoxPlot-Y-Values.pdf')
#
# a2_params = pd.unique(approach2['ParHAC_Threshold'])
# for model in a2_models:
# 	if model.split("/")[-1] in ['resnet18-cifar10.keras', 'resnet18-cifar100.keras', 'resnet18-svhn.keras']:
# 		continue
# 	plt.figure(figsize=(10, 6))
# 	Y = []
# 	for param in a2_params:
# 		a2_vals = approach2[approach2['ParHAC_Threshold'] == param][approach2['Model_Type'] == model]['MS_time']
# 		temp = []
# 		for index, thing in a2_vals.items():
# 			temp += [float(((avg_vanilla[model][2]) - thing) / (avg_vanilla[model][2]))]
# 		Y += [temp]
#
# 	plt.boxplot(Y,labels = a2_params)
#
# 	plt.xlabel('Parameters')
# 	plt.ylabel('Speedup')
# 	# plt.title(model.split("/")[-1]+' Box-and-Whisker Plot')
# 	plt.grid(True)
# 	plt.tight_layout()
# 	#
# 	# # Show plot
# 	plt.savefig('outputs/Approach2_'+model.split("/")[-1]+'_BoxPlot-Y-Values.pdf')


a2_params_man = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
#a2_params_man = [0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
a1_params_man = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
param_names = ['p1', 'p1', 'p2', 'p2', 'p3', 'p3', 'p4', 'p4', 'p5', 'p5', 'p6', 'p6', 'p7', 'p7', 'p8', 'p8', 'p9', 'p9', 'p10', 'p10']
models_bp = [value for value in a2_models if value in a1_models]
plot_type = [('MS_time', 2, 'MS_Time', 'Speedup')]#,('Mutation_Score', 1, 'MS', 'Mutation Score Error')]

for plot_type_col, dict_num, name, y_label in plot_type:
	for model in models_bp:
		if model.split("/")[-1] in ['resnet18-cifar10.keras', 'resnet18-cifar100.keras', 'resnet18-svhn.keras']:
			continue
		fig = plt.figure(figsize=(10, 6))
		Y = []
		for param_a1, param_a2 in zip(a1_params_man, a2_params_man):
			a1_vals = approach1[approach1['Neurons_per_Cluster_param'] == param_a1][approach1['Model_Type'] == model][plot_type_col]
			a2_vals = approach2[approach2['ParHAC_Threshold'] == param_a2][approach2['Model_Type'] == model][plot_type_col]
			temp = []
			for index, thing in a1_vals.items():
				temp += [float(((avg_vanilla[model][dict_num]) - thing) / (avg_vanilla[model][dict_num]))]
			Y += [temp]
			temp = []
			for index, thing in a2_vals.items():
				temp += [float(((avg_vanilla[model][dict_num]) - thing) / (avg_vanilla[model][dict_num]))]
			Y += [temp]

		ax = fig.add_subplot(111)
		pos=[0.6,1.4,2.6,3.4,4.6,5.4,6.6,7.4,8.6,9.4,10.6,11.4,12.6,13.4,14.6,15.4,16.6,17.4,18.6,19.4]
		bp = ax.boxplot(Y,positions=pos, boxprops={'linewidth': 2}, patch_artist=True)
		ax.set_xticks([1, 3, 5, 7, 9, 11, 13, 15, 17, 19], ['p1','p2','p3','p4','p5','p6','p7','p8','p9','p10'])
		for i in range(1, 20, 2):
			if len(Y[i])!=0 and len(Y[i-1])!=0:
				U1, p = mannwhitneyu(Y[i-1], Y[i])
				#ax.annotate(str(p), xy=(i, 5.75))
				ax.text(i-0.5, max(max(Y[i-1]), max(Y[i]))*1.02, "{:.3f}".format(p), fontsize=16)
		# xticks = ax.xaxis.get_major_ticks()
		# for i in range(1, 20, 2):
		# 	xticks[i].label1.set_visible(False)

		plt.xlabel('Parameters', fontsize=24)
		plt.ylabel('Speedup', fontsize=24)
		# plt.title(model.split("/")[-1]+' Box-and-Whisker Plot for Mutation Testing Time')
		plt.grid(True)
		plt.tight_layout()

		#bp = ax.boxplot([data1, data2, data3], patch_artist=True)

		# Customize the colors
		colors = ['blue', 'red'] * 10
		for patch, color in zip(bp['boxes'], colors):
			patch.set_facecolor(color)

		colors = ['lightblue', 'orange'] * 10
		for median, color in zip(bp['medians'], colors):
			median.set_color(color)

		# # Show plot
		plt.savefig('outputs/Approach1-2_'+model.split("/")[-1]+'_BoxPlot-'+name+'.pdf')

		# print(Y)

#=======================================================================

a2_params_man = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
#a2_params_man = [0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
a1_params_man = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
param_names = ['p1', 'p1', 'p2', 'p2', 'p3', 'p3', 'p4', 'p4', 'p5', 'p5', 'p6', 'p6', 'p7', 'p7', 'p8', 'p8', 'p9', 'p9', 'p10', 'p10']
models_bp = [value for value in a2_models if value in a1_models]
plot_type = [('Mutation_Score', 1, 'Mutation_Score', 'Mutation Score Error')]#,('Mutation_Score', 1, 'MS', 'Mutation Score Error')]

for plot_type_col, dict_num, name, y_label in plot_type:
	for model in models_bp:
		if model.split("/")[-1] in ['resnet18-cifar10.keras', 'resnet18-cifar100.keras', 'resnet18-svhn.keras']:
			continue
		fig = plt.figure(figsize=(10, 6))
		Y = []
		for param_a1, param_a2 in zip(a1_params_man, a2_params_man):
			a1_vals = approach1[approach1['Neurons_per_Cluster_param'] == param_a1][approach1['Model_Type'] == model][plot_type_col]
			a2_vals = approach2[approach2['ParHAC_Threshold'] == param_a2][approach2['Model_Type'] == model][plot_type_col]
			temp = []
			for index, thing in a1_vals.items():
				temp += [float(((avg_vanilla[model][dict_num]) - thing) / (avg_vanilla[model][dict_num]))]
			Y += [temp]
			temp = []
			for index, thing in a2_vals.items():
				temp += [float(((avg_vanilla[model][dict_num]) - thing) / (avg_vanilla[model][dict_num]))]
			Y += [temp]

		ax = fig.add_subplot(111)
		pos=[0.6,1.4,2.6,3.4,4.6,5.4,6.6,7.4,8.6,9.4,10.6,11.4,12.6,13.4,14.6,15.4,16.6,17.4,18.6,19.4]
		bp = ax.boxplot(Y,positions=pos, boxprops={'linewidth': 2}, patch_artist=True)
		ax.set_xticks([1, 3, 5, 7, 9, 11, 13, 15, 17, 19], ['p1','p2','p3','p4','p5','p6','p7','p8','p9','p10'])
		for i in range(1, 20, 2):
			if len(Y[i])!=0 and len(Y[i-1])!=0:
				U1, p = mannwhitneyu(Y[i-1], Y[i])
				#ax.annotate(str(p), xy=(i, 5.75))
				ax.text(i-0.5, max(max(Y[i-1]), max(Y[i]))*1.02, "{:.3f}".format(p), fontsize=16)
		# xticks = ax.xaxis.get_major_ticks()
		# for i in range(1, 20, 2):
		# 	xticks[i].label1.set_visible(False)

		plt.xlabel('Parameters', fontsize=24)
		plt.ylabel('Mutation Score Error', fontsize=24)
		# plt.title(model.split("/")[-1]+' Box-and-Whisker Plot for Mutation Testing Time')
		plt.grid(True)
		plt.tight_layout()

		#bp = ax.boxplot([data1, data2, data3], patch_artist=True)

		# Customize the colors
		colors = ['blue', 'red'] * 10
		for patch, color in zip(bp['boxes'], colors):
			patch.set_facecolor(color)

		colors = ['lightblue', 'orange'] * 10
		for median, color in zip(bp['medians'], colors):
			median.set_color(color)

		# # Show plot
		plt.savefig('outputs/Approach1-2_'+model.split("/")[-1]+'_BoxPlot_msscore.pdf')

#----------------------------------------------------------------------------------------------
# Mann-Whitney U-test Approach1/2 for both Mutation Test and Time and in between parameters for each model
# # U1, p = mannwhitneyu()
# P-value significantly smaller than 0.05 will indicate/signify statistical significance with a confidence level of 95%
'''
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
# plt.savefig('outputs/MannWhitneyUTestValues.pdf')'''


#======================================================================================================================================


label = []
X = []
Y = []
a2_params_man = [0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
a2_models = pd.unique(approach2['Model_Type'])

for model in a2_models:
	if model.split("/")[-1] in ['resnet18-cifar10.keras', 'resnet18-cifar100.keras', 'resnet18-svhn.keras']:
		continue
	temp = []
	tempy = []
	for param in a2_params_man:
		temp += [param]
		tempy += [(approach2[approach2['ParHAC_Threshold'] == param][approach2['Model_Type'] == model]['Number_of_Clusters']).mean()]

	Y += [tempy]
	X += [temp]
	label += [model.split('/')[-1].split('.')[0]]
X = np.array(X)
Y = np.array(Y)
labels = np.array(label)

# Plotting
plt.figure(figsize=(10, 6))

for x, y, l in zip(X, Y, labels):
	plt.plot(x, y, label=l)
# Customizing the plot
plt.xticks(np.array(a2_params))
plt.xlabel('PARHAC Threshold')
plt.ylabel('Number of Clusters')
plt.legend(title='Models')
plt.grid(True)
plt.tight_layout()

# Show plot
plt.savefig('outputs/Approach2_NumofClusters-vs-PARHAC.pdf')

#===========================================================================================================================================


label_a1 = []
label_a2 = []
A1_Y = []
A2_Y = []
X = []
#a2_params_man = [0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
a2_params_man = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
a1_params_man = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
a2_models = pd.unique(approach2['Model_Type'])
a1_models = pd.unique(approach1['Model_Type'])
param_names = ['p1', 'p2','p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10']
models_both = [value for value in a2_models if value in a1_models]
models = [[m for m in models_both if 'fcnn' in m], [m for m in models_both if 'lenet5' in m]]

for model in models:
	# want fcnn line for approach 1 and 2
	# want lenet5 line for approach 1 and 2 (4 lines total)
	temp_a1_y = []
	temp_a2_y = []
	temp_x = []
	for param_a1, param_a2, param in zip(a1_params_man, a2_params_man, param_names):
		a1_val = approach1[approach1['Neurons_per_Cluster_param'] == param_a1][approach1['Model_Type'].isin(model)]['Number_of_Clusters'].mean()
		a2_val = approach2[approach2['ParHAC_Threshold'] == param_a2][approach2['Model_Type'].isin(model)]['Number_of_Clusters'].mean()
		temp_a1_y += [a1_val]
		temp_a2_y += [a2_val]
		temp_x += [param]
	A1_Y += [temp_a1_y]
	A2_Y += [temp_a2_y]
	X += [temp_x]
	label_a1 += ['Neuron Clustering FCNN' if 'fcnn' in model[0] else 'Neuron Clustering LeNet-5']
	label_a2 += ['Mutant Clustering FCNN' if 'fcnn' in model[0] else 'Mutant Clustering LeNet-5']

A1_Y = np.array(A1_Y)
A2_Y = np.array(A2_Y)
X = np.array(X)
labels = np.array(label)
plt.figure(figsize=(10, 6))

for x, y, l in zip(X, A1_Y, label_a1):
	plt.plot(x, y, label=l)
for x, y, l in zip(X, A2_Y, label_a2):
	plt.plot(x, y, label=l)


# Customizing the plot
#plt.xticks(np.array(a2_params))
plt.xlabel('Parameter Value', fontsize=24)
plt.ylabel('Number of Clusters', fontsize=24)
plt.legend(title='Models', fontsize=22)
plt.grid(True)
plt.tight_layout()

# Show plot
plt.savefig('outputs/Approach1-2_NumofClusters-vs-Params.pdf')

# plt.figure(figsize=(10, 6))
# for x, y, l in zip(X, A2_Y, label_a2):
# 	plt.plot(x, y, label=l)
# plt.xlabel('Parameter Value', fontsize=24)
# plt.ylabel('Number of Clusters', fontsize=24)
# plt.legend(title='Models', fontsize=22)
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('outputs/Approach2_NumofClusters-vs-Params.pdf')
#===========================================================================================================================================

label_a1 = []
label_a2 = []
A1_Y = []
A2_Y = []
X = []
#a2_params_man = [0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
a2_params_man = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
a1_params_man = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
a2_models = pd.unique(approach2['Model_Type'])
a1_models = pd.unique(approach1['Model_Type'])
param_names = ['p1', 'p2','p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10']
models_both = [value for value in a2_models if value in a1_models]
models = [[m for m in models_both if 'fcnn' in m], [m for m in models_both if 'lenet5' in m]]

for model in models:
	# want fcnn line for approach 1 and 2
	# want lenet5 line for approach 1 and 2 (4 lines total)
	temp_a1_y = []
	temp_a2_y = []
	temp_x = []
	for param_a1, param_a2, param in zip(a1_params_man, a2_params_man, param_names):
		a1_val = approach1[approach1['Neurons_per_Cluster_param'] == param_a1][approach1['Model_Type'].isin(model)]['Number_of_Mutants'].mean()
		a2_val = approach2[approach2['ParHAC_Threshold'] == param_a2][approach2['Model_Type'].isin(model)]['Number_of_Clusters'].mean()
		temp_a1_y += [a1_val]
		temp_a2_y += [a2_val]
		temp_x += [param]
	A1_Y += [temp_a1_y]
	A2_Y += [temp_a2_y]
	X += [temp_x]
	label_a1 += ['FCNN Neuron Clustering' if 'fcnn' in model[0] else 'LeNet-5 Neuron Clustering']
	label_a2 += ['FCNN Mutant Clustering' if 'fcnn' in model[0] else 'LeNet-5 Mutant Clustering']

A1_Y = np.array(A1_Y)
A2_Y = np.array(A2_Y)
X = np.array(X)
labels = np.array(label)
plt.figure(figsize=(10, 6))

for x, y, l in zip(X, A1_Y, label_a1):
	plt.plot(x, y, label=l)
for x, y, l in zip(X, A2_Y, label_a2):
	plt.plot(x, y, label=l)

# plt.plot(['p1', 'p2','p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10'], [vanilla[vanilla['Model_Type'].isin(models[0])]['Number_of_Mutants'].mean()]*10, label='FCNN Vanilla Mutation Testing')
# plt.plot(['p1', 'p2','p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10'], [vanilla[vanilla['Model_Type'].isin(models[1])]['Number_of_Mutants'].mean()]*10, label='LeNet-5 Vanilla Mutation Testing')


# Customizing the plot
#plt.xticks(np.array(a2_params))
plt.xlabel('Parameter Value', fontsize=24)
plt.ylabel('Number of Mutants Tested', fontsize=24)
plt.legend(title='Models', fontsize=19, loc='center right', bbox_to_anchor=(1, 0.35))
plt.grid(True)
plt.tight_layout()

# Show plot
plt.savefig('outputs/Approach1-2_NumofTestedMutants-vs-Params.pdf')




from keras.models import load_model

models = ['fcnn-mnist.keras', 'lenet5-mnist.keras', 'fcnn-fmnist.keras', 'lenet5-fmnist.keras',
		  'fcnn-kmnist.keras', 'lenet5-kmnist.keras', 'fcnn-emnist.keras', 'lenet5-emnist.keras']

for m in models:
	s_model = load_model('examples/' + m.split('-')[1].split('.')[0] + '/' + m)
	print(m)
	print(s_model.summary())
	x_train = np.load('examples/' + m.split('-')[1].split('.')[0]  + '/data/' + m.split('-')[1].split('.')[0]  + '_train_inputs.npy')
	print(len(x_train))
	y_train = np.load('examples/' + m.split('-')[1].split('.')[0]  + '/data/' + m.split('-')[1].split('.')[0]  + '_train_outputs.npy')
	print(len(y_train))
	x_test = np.load('examples/' + m.split('-')[1].split('.')[0]  + '/data/' + m.split('-')[1].split('.')[0]  + '_test_inputs.npy')
	print(len(x_test))
	y_test = np.load('examples/' + m.split('-')[1].split('.')[0]  + '/data/' + m.split('-')[1].split('.')[0]  + '_test_outputs.npy')
	print(len(y_test))
	hist = s_model.evaluate(x_test, y_test)
	print(hist)
	# print('train_loss = ' + str(hist.history['loss']))
	# print('val_loss = ' + str(hist.history['val_loss']))
	# print('train_acc = ' + str(hist.history['acc']))
	# print('val_acc = ' + str(hist.history['val_acc']))
	# met = s_model.get_metrics_result()
	# print(met)
	print()