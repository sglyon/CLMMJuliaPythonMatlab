% FOC_VFI.m is a routine for computing capital satisfying the FOC; see 
% "Envelope Condition Method with an Application to Default Risk Models" 
% by Cristina Arellano, Lilia Maliar, Serguei Maliar and Viktor Tsyrennikov, 
% J Journal of Economic Dynamics and Control (2016), 69, 436-459 (henceforth, 
% AMMT, 2016).   
% -------------------------------------------------------------------------
% Inputs:    "k1" is next-period capital;
%            "k0" is current capital;
%            "z0" is current productivity;
%            "A" is the normalizing constant in production;
%            %alpha" is the share of capital in production;
%            "gam" is the utility-function parameter;
%            "delta" is the depreciation rate;
%            "beta" is the discount factor; 
%            "z1" is current level of productivity;
%            "n_nodes" is the integration nodes;
%            "weight_nodes" is the integration weights;
%            "vf" is the vector of polynomial coefficients in the approximated 
%               value function;  
%            "D" is the degree of polynomial approximation
%
% Output:    "dif_FOC" is the difference between the right and left hand 
%            sides of the FOC 
% -------------------------------------------------------------------------
% Copyright © 2012-2016 by Lilia Maliar and Serguei Maliar. All rights 
% reserved. The code may be used, modified and redistributed under the 
% terms provided in the file "License_Agreement.txt".
% -------------------------------------------------------------------------


function [diff_FOC] = FOC_VFI(k1,k0,z0,A,alpha,gam,delta,beta,z1,n_nodes,weight_nodes,vf,D)

EVder = 0;                  % Initialize the expected derivative of value 
                            % function
for j = 1:n_nodes           % Sum up the expected derivative of value 
                            % function across the integration nodes
    X1der = Polynomial_deriv_2d([k1 z1(j)],D);  
                            % Construct the derivatives of basis functions 
                            % of polynomial approximating value function at
                            % integration node j
    EVder = EVder + X1der*vf*weight_nodes(j); 
                            % Add up the weighted derivative of value 
                            % function across the nodes
end
diff_FOC = beta*EVder-((1-delta)*k0+A*z0*k0^alpha-k1)^(-gam); 
                            % Evaluate the difference between left and right
                            % sides of the FOC 
