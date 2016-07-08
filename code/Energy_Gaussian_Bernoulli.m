function [E] = Energy_Gaussian_Bernoulli(av,ah,net)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           ENERGY (GAUSSIAN-BERNOULLI)
% Compute Energy for a 2 layer network with GAUSSIAN to
% Logistic activation. The network is made up of input 
% and output layers only.
%
% INPUT : net -- The 2 layer network. This is a structure array (struct)
%                that holds information about the weights of the network.       
%         av  -- Input activation
%         ah  -- Output activation
%
% OUTPUT : E  -- Energy with respect to av, ah, and net.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Compute the BAM enregy Gaussian_Bernoulli
E1 = av'*net.W*ah;
E2 = 0.5* sum((av'-net.Bias2).^2);
E3 = ah'*net.Bias1;
E = -E1 + E2 - E3;
end

