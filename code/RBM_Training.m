clear;
%% Load the Input Data
load X;
N = size(X,1);

%% Define optimization parameters.  
% Define the optimization parameter

opts.learning = 0.5;
opts.numepochs = 200;

%---Batch number must be less than the number of input samples.
opts.batch = 10;

%% Define the Neural Network

%---Set the size of input neruons.
I = size(X,2);

%---Set number of output neurons to any desired value.
J = 5;

%% Initialize the neural network

nn.layers = {
    struct('type', 'input', 'size', I)
    struct('type', 'output','size', J)
    };

%--- Initialize the weights
% Weight and bias random range
e = 1.2;
b = -e;
opts.e = e;

%---Initialize the weights 
nn.W = unifrnd(b, e,nn.layers{2}.size,nn.layers{1}.size);

%---Initiialize Bias Weight
nn.Bias1 = unifrnd(b, e, nn.layers{2}.size,1);
nn.Bias2 = unifrnd(b, e, nn.layers{1}.size,1);

%% Set the activation  functions and network mode.

%---Input activation can be any of {'Sigmoid', 'Gaussian'}
% Default activation is Sigmoid
nn.layers{1}.fxn = 'Sigmoid';

% Set the output activation. It can only be 'Sigmoid'.
nn.layers{2}.fxn = 'Sigmoid';

%---Set the energy mode. 
% Default mode is 'Bernoulli_Bernoulli'.
nn.mode = 'Bernoulli_Bernoulli';
if (strcmp(nn.layers{1}.fxn, 'Gaussian'))
    nn.mode = 'Gaussian_Bernoulli';
end
%% Save the network information to result.txt

fileID = fopen('result.txt','w');
fprintf(fileID, 'Number of Input Neurons: %d \n', nn.layers{1}.size);
fprintf(fileID, 'Input activation function: %s \n', nn.layers{1}.fxn);
fprintf(fileID, 'Number of Output Neurons: %d \n', nn.layers{2}.size);
fprintf(fileID, 'Output activation function: %s \n', nn.layers{2}.fxn);
fprintf(fileID, 'Network Energy Mode: %s \n', nn.mode);
fprintf(fileID, 'Learning Rate: %2.3f \n', opts.learning);
fprintf(fileID, 'Number of Iterations: %d \n', opts.numepochs);
fprintf(fileID, '\n Weights before training is:\n');
dlmwrite('result.txt',nn.W, '-append','precision','%8.3f');
fclose(fileID);
%% NETWORK TRAINING WITH STOCHASTIC GRADIENT DESCENT.

for iter = 1:1:opts.numepochs
    
   %--- Randomly select batch input set.  
   OV = X(randsample(opts.batch, N),:); 
   
   % Compute activation for batch input set 
   AV = Activation_Fxn(nn.layers{1}.fxn, OV);
   
   %--Forward Pass to Output Layer
   OH = (AV * nn.W')+ repmat(nn.Bias1', N, 1);
   AH = Activation_Fxn(nn.layers{2}.fxn, OH);
   
   %-- Backward Pass to Input Layer
   OV = (AH * nn.W) + repmat(nn.Bias2', N, 1);
   AV = Activation_Fxn(nn.layers{1}.fxn, OV);
   
   %--Compute energy for each pair of (av, ah)
   E = zeros(size(AV,1),size(AH,1));
   for m = 1:1:size(AV,1)
       for  n = 1:1:size(AH,1)
           if (strcmp(nn.mode,'Gaussian_Bernoulli'))
               E(m,n) = Energy_Gaussian_Bernoulli(AV(m,:)', AH(n,:)', nn);
           else
               E(m,n) = Energy_Bernoulli_Bernoulli(AV(m,:)', AH(n,:)', nn);
           end
       end
   end
   
   %--Compute the joint pdf p(x,h|W)
   Ztheta = sum(sum(exp(-E)));
   prob_xh = exp(-E) / Ztheta;
   
   %--Compute expectation over the pdf p(x,h| W) 
   E_xh = zeros(size(nn.W));
   for i = 1:1:nn.layers{1}.size
       for j = 1:1:nn.layers{2}.size
           ah_aj = AV(:,i)*AH(:,j)';
           E_xh(j,i) = sum(sum(prob_xh.*ah_aj));
       end
   end
   
   %--Compute conditional pdf p(h|x,W)
   prob_x_W = sum(prob_xh,2);
   prob_h_xW = bsxfun(@rdivide, prob_xh, prob_x_W);
   
   %-- Compute expectation over the pdf p(h|x,W)
   E_h_x = zeros(size(nn.W));
   for i = 1:1:nn.layers{1}.size
       for j = 1:1:nn.layers{2}.size
           A = AV(:,i) * AH(:,j)';
           P = A.*prob_h_xW; 
           E_h_x(j,i) = sum(sum(P)) / size(AV,1);
       end
   end
   
   %--Update the Weights
   del_W = E_h_x - E_xh;
   nn.W = nn.W + (opts.learning * del_W);
end

%% Save the result after training to result.txt
fileID = fopen('result.txt','a');
fprintf(fileID, '\n Weights after training is:\n');
dlmwrite('result.txt',nn.W, '-append', 'delimiter',' ', 'precision','%8.3f');
fclose(fileID);
