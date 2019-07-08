% NK_accuracy.m is a routine for evaluating accuracy of the solutions    
% to the new Keynesian model considerd in the article "Merging Simulation
% and Projection Approaches to Solve High-Dimensional Problems with an
% application to a New Keynesian Model" by Lilia Maliar and Serguei Maliar,
% Quantitative Economics 6, 1-47 (2015) (henceforth, MM, 2015).
% -------------------------------------------------------------------------
% Inputs:    "nua", "nuL", "nuR", "nuG", "nuB", "nuu" are the time series
%            of shocks generated with the innovations for the test;
%            "delta", "L", "Y", "Yn", "pie", "S", "F", "C" are the time
%            series solution generated with the innovations for test;
%            "rho_nua", "rho_nuL", "rho_nuR", "rho_nuu", "rho_nuB", "rho_nuG"
%            are the parameters of the laws of motion for shocks;
%            "mu", "gam", "epsil", "vartheta", "beta", "A", "tau", "rho",
%            "vcv", "beta", "phi_y", "phi_pie", "theta", "piestar" and "Gbar"
%            are the parameters of the model;
%            "vk_2d" is the matrix of coefficients of the GSSA solution;
%            "discard" is the number of data points to discard;
%            "Degree" is the degree of polynomial approximation
%
% Output:    "Residuals_mean" and "Residuals_max" are, respectively, the mean
%            and maximum absolute residuals across all points and all equi-
%            librium conditions; "Residuals_max_E" is the maximum absolute
%            residuals across all points, disaggregated  by optimality conditions;
%            "Residuals" are absolute residuals disaggregated by the equi-
%            librium conditions
% -------------------------------------------------------------------------
% Copyright © 2013-2016 by Lilia Maliar and Serguei Maliar. All rights reserved.
% The code may be used, modified and redistributed under the terms provided
% in the file "License_Agreement.txt".
% -------------------------------------------------------------------------

function [Residuals_mean, Residuals_max, Residuals_max_E, Residuals] = NK_accuracy(nua,nuL,nuR,nuG,nuB,nuu,R,delta,L,Y,Yn,pie,S,F,C,rho_nua,rho_nuL,rho_nuR,rho_nuu,rho_nuB,rho_nuG,gam,vartheta,epsil,beta,phi_y,phi_pie,mu,theta,piestar,vcv,discard,vk_2d,Gbar,zlb,Degree)

tic                     % Start counting time for running the test

[T] = size(nua,1);      % Infer the number of points on which accuracy is
                        % evaluated
Residuals = zeros(T,6); % Allocate memory to the matrix of residuals; T-by-6

% Integration method for evaluating accuracy
% ------------------------------------------
[n_nodes,epsi_nodes,weight_nodes] = Monomials_2(6,vcv);
                             % Monomial integration rule with 2N^2+1 nodes

% Compute variables on the given set of points
%---------------------------------------------

for t = 1:T;                 % For each given point,
        t;

   % Take the corresponding value for shocks at t
   %---------------------------------------------
   nuR0 = nuR(t,1); % nuR(t)
   nua0 = nua(t,1); % nua(t)
   nuL0 = nuL(t,1); % nuL(t)
   nuu0 = nuu(t,1); % nuu(t)
   nuB0 = nuB(t,1); % nuB(t)
   nuG0 = nuG(t,1); % nuG(t)

   % Compute shocks at t+1 in all future nodes using their laws of motion
   %---------------------------------------------------------------------
   % Note that we do not premultiply by standard deviations as epsi_nodes
   % already include them
   nuR1(1:n_nodes,1) = (ones(n_nodes,1)*nuR0)*rho_nuR + epsi_nodes(:,1);
   % nuR(t+1); n_nodes-by-1
   nua1(1:n_nodes,1) = (ones(n_nodes,1)*nua0)*rho_nua + epsi_nodes(:,2);
   % nua(t+1); n_nodes-by-1
   nuL1(1:n_nodes,1) = (ones(n_nodes,1)*nuL0)*rho_nuL + epsi_nodes(:,3);
   % nuL(t+1); n_nodes-by-1
   nuu1(1:n_nodes,1) = (ones(n_nodes,1)*nuu0)*rho_nuu + epsi_nodes(:,4);
   % nuu(t+1); n_nodes-by-1
   nuB1(1:n_nodes,1) = (ones(n_nodes,1)*nuB0)*rho_nuB + epsi_nodes(:,5);
   % nuB(t+1); n_nodes-by-1
   nuG1(1:n_nodes,1) = (ones(n_nodes,1)*nuG0)*rho_nuG + epsi_nodes(:,6);
   % nuG(t+1); n_nodes-by-1

   R0  = R(t,1);            % R(t-1)
   delta0  = delta(t,1);    % delta(t-1)
   R1  = R(t+1,1);          % R(t)
   delta1  = delta(t+1,1);  % delta(t)

   L0 = L(t,1);             % L(t)
   Y0 = Y(t,1);             % Y(t)
   Yn0 = Yn(t,1);           % Yn(t)
   pie0 = pie(t,1);         % pie(t)
   S0 = S(t,1);             % S(t)
   F0 = F(t,1);             % F(t)
   C0 = C(t,1);             % C(t)

   % Future choices at t+1
   %----------------------
   delta1_dupl = ones(n_nodes,1)*delta1;
   R1_dupl = ones(n_nodes,1)*R1;
   % Duplicate "delta1" and "R1" n_nodes times to create a matrix with
   % n_nodes identical rows; n_nodes-by-1

   X1 = Ord_Polynomial_N([log(R1_dupl) log(delta1_dupl) nuR1 nua1 nuL1 nuu1 nuB1 nuG1],Degree);
   % Form a complete polynomial of degree "Degree" (at t+1) on future state
   % variables; n_nodes-by-npol_2d

   S1 = X1*vk_2d(:,1);             % Compute S(t+1) in all nodes using vk_2d
   F1 = X1*vk_2d(:,2);             % Compute F(t+1) in all nodes using vk_2d
   C1 = (X1*vk_2d(:,3)).^(-1/gam); % Compute C(t+1) in all nodes using vk_2d
   pie1 = ((1-(1-theta)*(S1./F1).^(1-epsil))/theta).^(1/(epsil-1));
                                   % Compute pie(t+1) using condition (35)
                                   % in MM (2015)

    % Compute residuals for each of the 9 equilibrium conditions
    %-----------------------------------------------------------
    Residuals(t,1) = 1-weight_nodes'*(exp(nuu0)*exp(nuL0)*L0^vartheta*Y0/exp(nua0) + beta*theta*pie1.^epsil.*S1)/S0;
    Residuals(t,2) = 1-weight_nodes'*(exp(nuu0)*C0^(-gam)*Y0 + beta*theta*pie1.^(epsil-1).*F1)/F0;
    Residuals(t,3) = 1-weight_nodes'*(beta*exp(nuB0)/exp(nuu0)*R1*exp(nuu1).*C1.^(-gam)./pie1)/C0^(-gam);
    Residuals(t,4) = 1-((1-theta*pie0^(epsil-1))/(1-theta))^(1/(1-epsil))*F0/S0;
    Residuals(t,5) = 1-((1-theta)*((1-theta*pie0^(epsil-1))/(1-theta))^(epsil/(epsil-1)) + theta*pie0^epsil/delta0)^(-1)/delta1;
    Residuals(t,6) = 1-exp(nua0)*L0*delta1/Y0;
    Residuals(t,7) = 1-(1-Gbar/exp(nuG0))*Y0/C0;
    Residuals(t,8) = 1-(exp(nua0)^(1+vartheta)*(1-Gbar/exp(nuG0))^(-gam)/exp(nuL0))^(1/(vartheta+gam))/Yn0;
    Residuals(t,9) = 1-piestar/beta*(R0*beta/piestar)^mu*((pie0/piestar)^phi_pie * (Y0/Yn0)^phi_y)^(1-mu)*exp(nuR0)/R1;   % Taylor rule
    if (zlb==1); Residuals(t,9) = Residuals(t,9)*(R1>1);end
    % If the ZLB is imposed and R>1, the residuals in the Taylor rule (the
    % 9th equation) are zero

end

% Residuals across all the equilibrium conditions and all test points
%--------------------------------------------------------------------
   Residuals_mean = log10(mean(mean(abs(Residuals(1+discard:end,:)))));
   % Mean absolute residuals computed after discarding the first
   % "discard" observations

   Residuals_max = log10(max(max(abs(Residuals(1+discard:end,:)))));
   % Maximum absolute residuals computed after discarding the first
   % "discard" observations

% Residuals disaggregated by the eqiulibrium conditions
%------------------------------------------------------
 Residuals_max_E = log10(max(abs(Residuals(1+discard:end,:))))';
   % Maximum absolute residuals across all test points for each of the 9
   % equilibrium conditions computed after discarding the first "discard"
   % observations;

time_test = toc;     % Time needed to run the test
