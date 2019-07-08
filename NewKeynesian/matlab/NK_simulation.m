% NK_simulation.m is a routine for simulating the solution to the new
% Keynesian model considerd in the article "Merging Simulation and Projection
% Approaches to Solve High-Dimensional Problems with an application to a New
% Keynesian Model" by Lilia Maliar and Serguei Maliar, Quantitative Economics
% 6, 1-47 (2015) (henceforth, MM, 2015).
% -------------------------------------------------------------------------

% -------------------------------------------------------------------------
% Inputs:    "vk" is the matrix of coefficients of the GSSA solution;
%            "nua", "nuL", "nuR", "nuG", "nuB", "nuu" are the time series
%            of shocks generated with the innovations for the test;
%            "R_init" and "delta_init" are initial values for R and delta
%            "gam", "vartheta", "epsil", "betta", "phi_y", "phi_pie", "mu",
%            "theta", "piestar", "Gbar"  are the parameters of the model;
%            "zlb" is a dummy parameter which is equal to 0 when ZLB is not
%            imposed and is equal to 1 when it is imposed;
%            "Degree" is the degree of polynomial approximation
%
% Output:    "S", "F", "delta", "C", "Y", "Yn", "L", "R" and "pie2" are the
%             simulated series
% -------------------------------------------------------------------------
% Copyright � 2013-2016 by Lilia Maliar and Serguei Maliar. All rights reserved.
% The code may be used, modified and redistributed under the terms provided
% in the file "License_Agreement.txt".
% -------------------------------------------------------------------------


function [S F delta C Y Yn L R pie w] = NK_simulation(vk,nuR,nua,nuL,nuu,nuB,nuG,R_init,delta_init,gam,vartheta,epsil,betta,phi_y,phi_pie,mu,theta,piestar,Gbar,zlb,Degree)


[T] = size(nua,1);       % Infer the number of points on which the accuracy
                         % is evaluated

delta = ones(T+1,1);     % Allocate memory for the time series of delta(t)
R = ones(T+1,1);         % Allocate memory for the time series of R(t)
S = ones(T,1);           % Allocate memory for the time series of S(t)
F = ones(T,1);           % Allocate memory for the time series of F(t)
C = ones(T,1);           % Allocate memory for the time series of C(t)
pie = ones(T,1);         % Allocate memory for the time series of pie(t)
Y = ones(T,1);           % Allocate memory for the time series of Y(t)
L = ones(T,1);           % Allocate memory for the time series of L(t)
Yn = ones(T,1);          % Allocate memory for the time series of Yn(t)
w  = ones(T,1);

delta(1,1) = delta_init; % Initial condition for delta(t-1), i.e., delta(-1)
R(1,1) = R_init;         % Initial condition for R(t-1)

for t = 1:T
    pol_bases = Ord_Polynomial_N([log(R(t,1)) log(delta(t,1)) nuR(t,1) nua(t,1) nuL(t,1) nuu(t,1) nuB(t,1) nuG(t,1)],Degree);
    % Construct the matrix of explanatory variables "pol_bases" on the series
    % of state variables; columns of "pol_bases" are given by the basis
    % functions of the polynomial of degree "Degree"
    S(t,1) = pol_bases*vk(:,1);             % Compute S(t) using vk
    F(t,1) = pol_bases*vk(:,2);             % Compute F(t) using vk
    C(t,1) = (pol_bases*vk(:,3)).^(-1/gam); % Compute C(t) using vk
    pie(t,:) = ((1-(1-theta)*(S(t,:)/F(t,:))^(1-epsil))/theta)^(1/(epsil-1));
                         % Compute pie(t) from condition (35) in MM (2015)
    delta(t+1,:) = ((1-theta)*((1-theta*pie(t,1)^(epsil-1))/(1-theta))^(epsil/(epsil-1))+theta*pie(t,1)^epsil/delta(t,1))^-1;
                          % Compute delta(t) from condition (36) in MM (2015)
    Y(t,:) = C(t,1)/(1-Gbar/exp(nuG(t,1)));
                         % Compute Y(t) from condition (38) in MM (2015)
    L(t,1) = Y(t,1)/delta(t+1,1)/exp(nua(t,1));
                         % Compute L(t) from condition (37) in MM (2015)
    Yn(t,:) = (exp(nua(t,1))^(1+vartheta)*(1-Gbar/exp(nuG(t,1)))^(-gam)/exp(nuL(t,1)))^(1/(vartheta+gam));
                         %  Compute Yn(t) from condition (31) in MM (2015)
    R(t+1,:) = piestar/betta*(R(t,:)*betta/piestar)^mu*((pie(t,:)/piestar)^phi_pie*(Y(t,:)/Yn(t,:))^phi_y)^(1-mu)*exp(nuR(t,1));
         % Compute R(t) from conditions (27), (39) in MM (2015)
    w(t,1) = exp(nuL(t,1))*(L(t,1)^vartheta)*(C(t,1)^gam);
         % Compute real wage

    if zlb == 1
        R(t+1,:) = max(R(t+1,:),1);
        % If ZLB is imposed, set R(t)=1 if ZLB binds
    end;

end
