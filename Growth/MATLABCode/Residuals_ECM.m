% Residuals_ECM.m is a routine for evaluating the accuracy of the solutions
% to the neoclassical stochastic growth model; it computes approximation  
% residuals in the optimality conditions on a given set of points in the  
% state space; see "Envelope Condition Method with an Application to Default 
% Risk Models" by Cristina Arellano, Lilia Maliar, Serguei Maliar and Viktor 
% Tsyrennikov, Journal of Economic Dynamics and Control (2016), 69, 436-459 
% (henceforth, AMMT, 2016).    
% -------------------------------------------------------------------------

% -------------------------------------------------------------------------
% Inputs:    "k" is current capital;
%            "z" is current productivity
%            "A" is the normalizing constant in production;
%            "alpha" is the share of capital in production;
%            "gam" is the utility-function parameter;
%            "delta" is the depreciation rate;
%            "sigma" is the standard deviation of the productivity shock;
%            "rho" is the autocorrelation coefficient in the process for
%                shock;
%            "beta" is the discount factor; 
%            "vf" is the vector of polynomial coefficients in the 
%               approximated value function;  
%            "D" is the degree of polynomial approximation
%
% Output:    "Mean_Residuals" and "Max_Residuals" are, respectively, the 
%            mean and maximum absolute Euler equation residuals (in log10)
% -------------------------------------------------------------------------
% Copyright © 2012-2016 by Lilia Maliar and Serguei Maliar. All rights 
% reserved. The code may be used, modified and redistributed under the 
% terms provided in the file "License_Agreement.txt".
% -------------------------------------------------------------------------

function [Mean_Residuals,Max_Residuals]  = Residuals_ECM(k,z,A,alpha,gam,delta,sigma,rho,beta,vf,D)

T     = size(k,1);   % Simulation length

% Gauss Hermite quadrature integration in the accuracy evaluation procedure:
% the number of integration nodes, their values and weights
%--------------------------------------------------------------------------
Qn = 10;             % Number of unidimensional integration nodes 
nshocks = 1;         % There is one stochastic shock
vcv = sigma^2;       % Variance covariance matrix
[n_nodes,epsi_nodes,weight_nodes] = GH_Quadrature(Qn,nshocks,vcv);
                     % Gauss Hermite quadrature nodes and weights;
                     % with one shock, n_nodes = Qn
                     
 for t=1:T;   
     
     k0 = k(t,1);         % Capital of period t
     z0 = z(t,1);         % Productivity level of period t 
             
     % Current period quantities in (k0,a0)
     % ------------------------------------     
     Vder0= Polynomial_deriv_2d([k0 z0],D)*vf;   % Construct derivative of 
                                                 % value function
     u0=Vder0./(1-delta+A*alpha*z0*k0^(alpha-1));% Envelope condition
     c0=u0.^(-1/gam);                            % Consumption
     k1=(1-delta)*k0+A*z0.*k0^alpha-c0;          % Next-period capital
     
     % Future period quantities in n_nodes integration nodes (k1,a1)
     % --------------------------------------------------------     
     z1 = z0.^rho.*exp(epsi_nodes); % Productivity in n_nodes nodes
     k1_dupl = ones(n_nodes,1)*k1;  % k1 is the same in n_nodes nodes       
     Vder1(1:n_nodes,1) =  Polynomial_deriv_2d([k1_dupl z1(:,1)],D)*vf;   
                                    % Construct next period derivative
                                    % of value function in the integration
                                    % nodes
     u1=Vder1./(1-delta+A*alpha*z1(:,1).*k1.^(alpha-1)); 
                                    % Envelope condition
     c1(1:n_nodes,1)=u1.^(-1/gam);  % Next period consumption
  
     % Unit-free residuals in the Euler equation
     % -----------------------------------------
     Residuals(t,1) = weight_nodes'*(beta*c1(1:n_nodes,1).^(-gam)./c0.^(-gam).*(1-delta+alpha*A*z1(1:n_nodes,1).*k1.^(alpha-1)))-1;

 end

 % Output of the accuracy evaluation test
 % --------------------------------------
 Mean_Residuals = log10(mean(mean(abs(Residuals(1:end))))); % Mean residuals 
 Max_Residuals = log10(max(max(abs(Residuals(1:end)))));    % Max  residuals
    
 
    