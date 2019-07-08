% EGM_BC.m is a routine for computing current capital satisfying the budget 
% constraint; see  "Envelope Condition Method with an Application to Default 
% Risk Models" by Cristina Arellano, Lilia Maliar, Serguei Maliar and Viktor 
% Tsyrennikov, Journal of Economic Dynamics and Control (2016), 69, 436-459 
% (henceforth, AMMT, 2016).   
% -------------------------------------------------------------------------
% Inputs:    "k1" is next-period capital;
%            "k0" is current capital;
%            "c0" is current consumption;
%            "z0" is current productivity;
%            "A" is the normalizing constant in production;
%            %alpha" is the share of capital in production;
%            "delta" is the depreciation rate;
%
% Output:    "dif_BC" is the difference between the right and left hand sides 
%            of the budget constraint 
% -------------------------------------------------------------------------
% Copyright © 2012-2016 by Lilia Maliar and Serguei Maliar. All rights 
% reserved. The code may be used, modified and redistributed under the 
% terms provided in the file "License_Agreement.txt".
% -------------------------------------------------------------------------


function [diff_BC] = EGM_BC(k0,k1,z0,c0,A,alpha,delta)

diff_BC = (1-delta)*k0+A*z0.*k0.^alpha-k1-c0; 
                            % Evaluate the difference between left and right
                            % sides of the budget constraint 
