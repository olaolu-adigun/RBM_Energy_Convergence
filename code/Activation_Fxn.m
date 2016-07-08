function [a] = Activation_Fxn(fxn_name, o)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                       ACTIVATION FUNCTION
% Compute the activation for an input vector. The possible activations are:
%          -- Gaussain
%          -- Sigmoid
%
% INPUT : 
%  fxn_name -- This specifies the desired activation function. This 
%              can be any of the functions above. The default activtaion 
%              is Sigmoid function. 
%         o -- Input vector
% OuTPUT:
%         a -- Activation vector for input vetor o.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Compute activation vector

if (strcmp(fxn_name, 'Gaussian'))
    a = exp(-1*(o.^2));
else
    a = (1./(1 + exp(-1*o)));
end

end

