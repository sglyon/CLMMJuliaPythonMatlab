% VF_Bellman.m is a routine for computing value function on the left side of
% the Bellman equation; see "Envelope Condition Method with an Application 
% to Default Risk Models" by Cristina Arellano, Lilia Maliar, Serguei Maliar 
% and Viktor Tsyrennikov,  Journal of Economic Dynamics and Control (2016), 
% 69, 436-459 (henceforth, AMMT, 2016).   
% -------------------------------------------------------------------------

% -------------------------------------------------------------------------
% Inputs:    "c0" is current consumption;
%            "k1" is next-period capital;
%            "z1" is future productivity;
%            "gam" is the utility-function parameter;
%            "beta" is the discount factor; 
%            "n_nodes" is the number of integration nodes;
%            "weight_nodes" is the integration weights;
%            "vf_coef" is the vector of polynomial coefficients in the 
%               approximated value function;  
%            "D" is the degree of polynomial approximation
%
% Output:    "V_new" is value function on the left side of the Bellman
%            equation
% -------------------------------------------------------------------------
% Copyright © 2012-2016 by Lilia Maliar and Serguei Maliar. All rights 
% reserved. The code may be used, modified and redistributed under the 
% terms provided in the file "License_Agreement.txt".
% -------------------------------------------------------------------------

function [V_new] = VF_Bellman(c0,k1,z1,gam,beta,n_nodes,weight_nodes,vf_coef,D)


for j = 1:n_nodes                         % For each integration node...
    X1 = Polynomial_2d([k1 z1(:,j)],D);   % Construct polynomial   
    EV(:,j) = X1*vf_coef;                 % Evaluate value function
end

if gam==1
    V_new = log(c0)+beta*EV*weight_nodes; % Bellman equation if gam=1        
else
    V_new = (c0.^(1-gam)-1)/(1-gam)+beta*EV*weight_nodes;
end                                       % Bellman equation otherwise