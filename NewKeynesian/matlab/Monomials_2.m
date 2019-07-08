% Monomials_2.m is a routine that constructs integration nodes and weights 
% under N-dimensional monomial (non-product) integration rule with 2N^2+1 
% nodes; see Judd, Maliar and Maliar, (2010), "A Cluster-Grid Projection 
% Method: Solving Problems with High Dimensionality", NBER Working Paper 
% 15965 (henceforth, JMM, 2010).
% -------------------------------------------------------------------------
% Inputs:  "N" is the number of random variables; N>=1;
%          "vcv" is the variance-covariance matrix; N-by-N;

% Outputs: "n_nodes" is the total number of integration nodes; 2*N^2+1;
%          "epsi_nodes" are the integration nodes; n_nodes-by-N;
%          "weight_nodes" are the integration weights; n_nodes-by-1
% -------------------------------------------------------------------------
% Copyright © 2011 by Lilia Maliar and Serguei Maliar. All rights reserved. 
% The code may be used, modified and redistributed under the terms provided 
% in the file "License_Agreement.txt".
% -------------------------------------------------------------------------


function [n_nodes,epsi_nodes,weight_nodes] = Monomials_2(N,vcv)

n_nodes = 2*N^2+1;    % Total number of integration nodes

% 1. N-dimensional integration nodes for N uncorrelated random variables with 
% zero mean and unit variance
% ---------------------------------------------------------------------------   

% 1.1 The origin point
% --------------------
z0 = zeros(1,N);       % A supplementary matrix for integration nodes: the 
                       % origin point 

% 1.2 Deviations in one dimension
% -------------------------------
z1 = zeros(2*N,N);     % A supplementary matrix for integration nodes; 
                       % n_nodes-by-N
                       
for i = 1:N            % In each node, random variable i takes value either
                       % 1 or -1, and all other variables take value 0
    z1(2*(i-1)+1:2*i,i) = [1; -1];  
end                    % For example, for N = 2, z1 = [1 0; -1 0; 0 1; 0 -1]

% 1.3 Deviations in two dimensions
% --------------------------------
z2 = zeros(2*N*(N-1),N);   % A supplementary matrix for integration nodes; 
                           % 2N(N-1)-by-N

i=0;                       % In each node, a pair of random variables (p,q)
                           % takes either values (1,1) or (1,-1) or (-1,1) or    
                           % (-1,-1), and all other variables take value 0 
for p = 1:N-1                           
    for q = p+1:N      
        i=i+1;               
        z2(4*(i-1)+1:4*i,p) = [1;-1;1;-1];    
        z2(4*(i-1)+1:4*i,q) = [1;1;-1;-1];
    end
end                 % For example, for N = 2, z2 = [1 1;1 -1;-1 1;-1 1]

% z = [z0;z1*sqrt(N+2);z2*sqrt((N+2)/2)];   % Integration nodes 
                       
% 2. N-dimensional integration nodes and weights for N correlated random 
% variables with zero mean and variance-covaraince matrix vcv 
% ----------------------------------------------------------------------  
sqrt_vcv = chol(vcv);            % Cholesky decomposition of the variance-
                                 % covariance matrix
                                 
R = sqrt(N+2)*sqrt_vcv;          % Variable R; see condition (21) in JMM 
                                 % (2010)
                                 
S = sqrt((N+2)/2)* sqrt_vcv;     % Variable S; see condition (21) in JMM 
                                 % (2010)
                                 
epsi_nodes = [z0;z1*R;z2*S]; 
                                 % Integration nodes; see condition (21)
                                 % in JMM (2010); n_nodes-by-N

% 3. Integration weights
%-----------------------
weight_nodes = [2/(N+2)*ones(size(z0,1),1);(4-N)/2/(N+2)^2*ones(size(z1,1),1);1/(N+2)^2*ones(size(z2,1),1)];
                                 % See condition (21) in JMM (2010); 
                                 % n_nodes-by-1; the weights are the same   
                                 % for the cases of correlated and 
                                 % uncorrelated random variables