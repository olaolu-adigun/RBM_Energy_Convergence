# RBM
CONTRASTIVE DIVERGENCE WITH BAMs AND RBMs (2-LAYER NETWORK).

This software trains BAMs and RBM network. The network energy mode can either be 'Bernoulli_Bernoulli' or 'Gaussian_Bernoulli'. The software  stochastic gradient
to update the weights during the training process.

INPUTS:
--- X    : Input Dataset saved as X.mat (There is dummy dataset in the folder).
--- opts : This is a structure array for optimization parameters. The parammeters are: 
         -- Learning rate
         -- Epoch size
         -- Batch size (Batch size shoould be less than total number of samples in the dataset).
--- J    : Size of output neurons.
--- nn   : This is a structure array for the network. The parameters are:
         -- Activation function for the input layer ('Gaussian' or 'Sigmoid'). The default input activation is 'Sigmoid.
         -- The output activation is always 'Sigmoid'.
         -- 
  
OUTPUTS
The software writes the following information to a result file:
-- Optimization Parameters
-- Network information       
-- Weights before training   
-- Weights after training

Instruction.
(1). Adjust the inputs paramters mentioned above to suit your purpose.
(2). Run the program by clicking the run button  under(RBM_Training.m).
(3). Open the output file (result.txt) to see the result.
