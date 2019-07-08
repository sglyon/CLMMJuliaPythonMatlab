% Monomials_1.m is a routine that constructs integration nodes and weights 
% under N-dimensional monomial (non-product) integration rule with 2N nodes; 
% see Judd, Maliar and Maliar, (2010), "A Cluster-Grid Projection Method: 
% Solving Problems with High Dimensionality", NBER Working Paper 15965 
% (henceforth, JMM, 2010).
% -------------------------------------------------------------------------
% Inputs:  "N" is the number of random variables; N>=1;
%          "vcv" is the variance-covariance matrix; N-by-N

% Outputs: "n_nodes" is the total number of integration nodes; 2*N;
%          "epsi_nodes" are the integration nodes; n_nodes-by-N;
%          "weight_nodes" are the integration weights; n_nodes-by-1
% -------------------------------------------------------------------------
% Copyright © 2011 by Lilia Maliar and Serguei Maliar. All rights reserved. 
% The code may be used, modified and redistributed under the terms provided 
% in the file "License_Agreement.txt".
% -------------------------------------------------------------------------

function [n_nodes,epsi_nodes,weight_nodes] = Monomials_1(N,vcv)

n_nodes   = 2*N;       % Total number of integration nodes

% 1. N-dimensional integration nodes for N uncorrelated random variables with 
% zero mean and unit variance
% ---------------------------------------------------------------------------   
z1 = zeros(n_nodes,N); % A supplementary matrix for integration nodes; 
                       % n_nodes-by-N
                       
for i = 1:N            % In each node, random variable i takes value either
                       % 1 or -1, and all other variables take value 0
    z1(2*(i-1)+1:2*i,i) = [1; -1];  
end                    % For example, for N = 2, z1 = [1 0; -1 0; 0 1; 0 -1]

% z = z1*sqrt(N);      % Integration nodes  

% 2. N-dimensional integration nodes and weights for N correlated random 
% variables with zero mean and variance-covaraince matrix vcv 
% ----------------------------------------------------------------------  
sqrt_vcv = chol(vcv);  % Cholesky decomposition of the variance-covariance 
                       % matrix
                                 
R = sqrt(N)*sqrt_vcv;  % Variable R; see condition (20) in JMM (2010)
                                 
epsi_nodes = z1*R;     % Integration nodes; see condition (20) in JMM (2010); 
                       % n_nodes-by-N

% 3. Integration weights
%-----------------------
weight_nodes = ones(n_nodes,1)/n_nodes; 
                       % Integration weights are equal for all integration 
                       % nodes; n_nodes-by-1; the weights are the same for 
                       % the cases of correlated and uncorrelated random 
                       % variables
