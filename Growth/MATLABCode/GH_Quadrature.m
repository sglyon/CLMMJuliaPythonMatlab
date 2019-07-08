% GH_Quadrature.m is a routine that constructs integration nodes and weights   
% under Gauss-Hermite quadrature (product) integration rule with Qn<=10  
% nodes in each of N dimensions; see the Supplement to "Numerically Stable  
% and Accurate Stochastic Simulation Approaches for Solving Dynamic Economic 
% Models" by Kenneth L. Judd, Lilia Maliar and Serguei Maliar, (2011),  
% Quantitative Economics 2/2, 173–210 (henceforth, JMM, 2011).
%
% This version: July 14, 2011. First version: August 27, 2009.
% -------------------------------------------------------------------------
% Inputs:  "Qn" is the number of nodes in each dimension; 1<=Qn<=10;
%          "N" is the number of random variables; N=1,2,...;
%          "vcv" is the variance-covariance matrix; N-by-N

% Outputs: "n_nodes" is the total number of integration nodes; Qn^N;
%          "epsi_nodes" are the integration nodes; n_nodes-by-N;
%          "weight_nodes" are the integration weights; n_nodes-by-1
% -------------------------------------------------------------------------
% Copyright © 2011 by Lilia Maliar and Serguei Maliar. All rights reserved. 
% The code may be used, modified and redistributed under the terms provided 
% in the file "License_Agreement.txt".
% -------------------------------------------------------------------------

function [n_nodes,epsi_nodes,weight_nodes] = GH_Quadrature(Qn,N,vcv)

% 1. One-dimensional integration nodes and weights (given with 16-digit 
% accuracy) under Gauss-Hermite quadrature for a normally distributed random 
% variable with zero mean and unit variance
% -------------------------------------------------------------------------
if Qn == 1                 % Number of nodes in each dimension; Qn <=10
    eps = [0];             % Set of integration nodes
    weight = [sqrt(pi)];   % Set of integration weights      
elseif Qn == 2;            
    eps = [0.7071067811865475; -0.7071067811865475]; 
    weight = [0.8862269254527580;  0.8862269254527580];
elseif Qn == 3;
    eps = [1.224744871391589; 0; -1.224744871391589];
    weight = [0.2954089751509193;1.181635900603677;0.2954089751509193];
elseif Qn == 4;
    eps = [1.650680123885785; 0.5246476232752903;-0.5246476232752903;-1.650680123885785];
    weight = [0.08131283544724518;0.8049140900055128; 0.8049140900055128; 0.08131283544724518];
elseif Qn == 5;
    eps = [2.020182870456086;0.9585724646138185;0;-0.9585724646138185;-2.020182870456086];
    weight = [0.01995324205904591;0.3936193231522412;0.9453087204829419;0.3936193231522412;0.01995324205904591];
elseif Qn == 6;
    eps = [2.350604973674492;1.335849074013697;0.4360774119276165;-0.4360774119276165;-1.335849074013697;-2.350604973674492];
    weight = [0.004530009905508846;0.1570673203228566;0.7246295952243925;0.7246295952243925;0.1570673203228566;0.004530009905508846];
elseif Qn == 7;
    eps = [2.651961356835233;1.673551628767471;0.8162878828589647;0;-0.8162878828589647;-1.673551628767471;-2.651961356835233];
    weight = [0.0009717812450995192; 0.05451558281912703;0.4256072526101278;0.8102646175568073;0.4256072526101278;0.05451558281912703;0.0009717812450995192]; 
elseif Qn == 8;
    eps = [2.930637420257244;1.981656756695843;1.157193712446780;0.3811869902073221;-0.3811869902073221;-1.157193712446780;-1.981656756695843;-2.930637420257244];
    weight = [0.0001996040722113676;0.01707798300741348;0.2078023258148919;0.6611470125582413;0.6611470125582413;0.2078023258148919;0.01707798300741348;0.0001996040722113676]; 
elseif Qn == 9;
    eps = [3.190993201781528;2.266580584531843;1.468553289216668;0.7235510187528376;0;-0.7235510187528376;-1.468553289216668;-2.266580584531843;-3.190993201781528];
    weight = [0.00003960697726326438;0.004943624275536947;0.08847452739437657;0.4326515590025558;0.7202352156060510;0.4326515590025558;0.08847452739437657;0.004943624275536947;0.00003960697726326438];    
else
    Qn =10; % The default option
    eps = [3.436159118837738;2.532731674232790;1.756683649299882;1.036610829789514;0.3429013272237046;-0.3429013272237046;-1.036610829789514;-1.756683649299882;-2.532731674232790;-3.436159118837738];
    weight = [7.640432855232621e-06;0.001343645746781233;0.03387439445548106;0.2401386110823147;0.6108626337353258;0.6108626337353258;0.2401386110823147;0.03387439445548106;0.001343645746781233;7.640432855232621e-06];
end

% 2. N-dimensional integration nodes and weights for N uncorrelated normally 
% distributed random variables with zero mean and unit variance
% ------------------------------------------------------------------------                        
n_nodes = Qn^N;        % Total number of integration nodes (in N dimensions)

z1 = zeros(n_nodes,N); % A supplementary matrix for integration nodes; 
                       % n_nodes-by-N 
w1 = ones(n_nodes,1);  % A supplementary matrix for integration weights; 
                       % n_nodes-by-1

for i = 1:N            
   z1i = [];           % A column for variable i to be filled in with nodes 
   w1i = [];           % A column for variable i to be filled in with weights 
   for j = 1:Qn^(N-i)
       for u=1:Qn
           z1i = [z1i;ones(Qn^(i-1),1)*eps(u)];
           w1i = [w1i;ones(Qn^(i-1),1)*weight(u)];
       end
   end
   z1(:,i) = z1i;      % z1 has its i-th column equal to z1i 
   w1 = w1.*w1i;       % w1 is a product of weights w1i 
end

z = sqrt(2).*z1;       % Integration nodes; n_nodes-by-N; for example, 
                       % for N = 2 and Qn=2, z = [1 1; -1 1; 1 -1; -1 -1]

w = w1/sqrt(pi)^N;     % Integration weights; see condition (B.6) in the 
                       % Supplement to JMM (2011); n_nodes-by-1

% 3. N-dimensional integration nodes and weights for N correlated normally 
% distributed random variables with zero mean and variance-covariance matrix, 
% vcv 
% -----------------------------------------------------------------------                      
sqrt_vcv = chol(vcv);            % Cholesky decomposition of the variance-
                                 % covariance matrix
                                 
epsi_nodes = z*sqrt_vcv;         % Integration nodes; see condition (B.6)  
                                 % in the Supplement to JMM (2011); 
                                 % n_nodes-by-N                                

weight_nodes = w;                % Integration weights are the same for the 
                                 % cases of correlated and uncorrelated 
                                 % random variables 