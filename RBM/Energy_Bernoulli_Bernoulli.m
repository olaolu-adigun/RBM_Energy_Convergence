function [E] = Energy_Bernoulli_Bernoulli(av,ah, net)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           ENERGY (BERNOULLI-BERNOULLI)
% Compute Energy for a 2 layer network with Logistic to
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


%% Compute the BAM enregy for Bernoulli-Bernoulli
E1 = ah'*net.W*av;
E2 = av'*net.Bias2;
E3 = ah'*net.Bias1;
E = -E1 - E2 - E3;
end

