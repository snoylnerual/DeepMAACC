# DeepMAACC
Deep Mutation Analysis Acceleration via Clustering


	Qdnn
	Cuda
	
	Run a Lenet5 on mnist
	Created conda environ
	Install Tensorfloefpu
	Install Keras
	
	See if I can use the gpu
	Device = -1
	Device = gpu0
	• 
	• 
	• Vanilla approach run it -> make it run in the background
	• Secondary hard drive
	• Tranfer those files onto my second hardrive

	Read the README file under graph-mining to understand how to build the library and obtain dll/so/dylib file. Also I have added comments in the files clustering.py and mutation_tester.py which might help you
	
	
	Ø Fix the mutant creation to make every mutant for each neuron that is changed (vanilla) and then for each cluster that is changed
	Ø Need to make it record where the mutation happens (layer and neuron number)
	Ø Then make a separate approach where we just use the vanilla mutation, but the clustering happens after. The clustering will happen for mutants based on similarity of location of mutation. You only cluster mutants on the same layer. You cluster the mutants on the eucclidian distance of weights and biases combined. You will give a tuple of the location and also weights and biases to the eucclidian distance. There is something in scipy that is helpful for eucclidian distance
	Ø Use the clustering or whatever from the dms codebase that he sent to me in an email.
	Ø Then for testing, you test one representative from the mutant clusters and if that one is killed/kills a class, then all are killed.
	Ø Each mutation operator changes one neuron
	Ø CHANGED CLUSTERS PER LAYER TO CLUSTER SIZE

If you were to use the clusters, there is a chance that most clusters will hold a different number of neurons. This would mean there could be dimension mismatches when trying to calculate the eucclidian distance.


-gets model.


How can I make it where the layers that actually have clusters with them are the only ones that are edited?
	- In the mutation generators, the cluster has a layer number associated with it, so you should be able to use that to get the right layers without the specific numbers being held somewhere… but what do you do about normal mutation? 
	





Taken incite and dm and put them together to check mutation analysis stuff like comparing the time it takes for them to compute vanilla and clustered.

Research Questions

How much speedup you gain when using neuron clustering vs vanilla mutation testing? 

How much mutation score is lost when using neuron clustering vs vanilla mutation testing? 

What is the impact of non-determinacy in training process on mutation score of your tool when using neuron clustering vs in vanilla mode

https://dl.acm.org/doi/10.1145/2483760.2483782

What deepMAACC paper should end up like
![image](https://github.com/user-attachments/assets/cfca73e4-fae4-43c2-85ad-9fdc9fcfa2b6)
