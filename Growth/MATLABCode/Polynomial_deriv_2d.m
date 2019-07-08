% Polynomial_deriv_2d.m is a routine for constructing the derivatives of 
% basis functions of complete ordinary polynomial of the degrees from one 
% to five for the two-dimensional case; see  "Envelope Condition Method with 
% an Application to Default Risk Models" by Cristina Arellano, Lilia Maliar, 
% Serguei Maliar and Viktor Tsyrennikov,  Journal of Economic Dynamics and 
% Control (2016), 69, 436-459 (henceforth, AMMT, 2016).   
% -------------------------------------------------------------------------

% -------------------------------------------------------------------------
% Inputs:    "s" is the data points on which the derivatives of the poly-  
%            nomial basis functions must be constructed; T-by-2;
%            "D" is the degree of the polynomial whose basis functions must 
%            be constructed; (can be 1,2,3,4 or 5)
%
% Output:    "x" is the matrix of the derivatives of basis functions of
%            complete polynomial of the given degree with respect to
%            first argument
% -------------------------------------------------------------------------
% Copyright © 2012-2016 by Lilia Maliar and Serguei Maliar. All rights 
% reserved. The code may be used, modified and redistributed under the 
% terms provided in the file "License_Agreement.txt".
% -------------------------------------------------------------------------


function x = Polynomial_deriv_2d(s,D)

T = size(s,1);  % s can be a scalar or vector 

p1 = s(:,1); p2 = s(:,1).^2; p3 = s(:,1).^3; p4 = s(:,1).^4;
q1 = s(:,2); q2 = s(:,2).^2; q3 = s(:,2).^3; q4 = s(:,2).^4;    

if D==1;
    x = [zeros(T,1) ones(T,1) zeros(T,1)];
elseif D==2;
    x = [zeros(T,1) ones(T,1) zeros(T,1) 2*p1 q1 zeros(T,1)];
elseif D==3;
    x = [zeros(T,1) ones(T,1) zeros(T,1) 2*p1 q1 zeros(T,1) 3*p2 2*p1.*q1 q2 zeros(T,1)];
elseif D==4;
    x = [zeros(T,1) ones(T,1) zeros(T,1) 2*p1 q1 zeros(T,1) 3*p2 2*p1.*q1 q2 zeros(T,1) 4*p3 3*p2.*q1 2*p1.*q2 q3 zeros(T,1)];
elseif D==5;
    x = [zeros(T,1) ones(T,1) zeros(T,1) 2*p1 q1 zeros(T,1) 3*p2 2*p1.*q1 q2 zeros(T,1) 4*p3 3*p2.*q1 2*p1.*q2 q3 zeros(T,1) 5*p4 4*p3.*q1 3*p2.*q2 2*p1.*q3 q4 zeros(T,1)];
end    
