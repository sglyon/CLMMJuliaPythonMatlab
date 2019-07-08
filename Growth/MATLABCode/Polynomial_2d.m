% Polynomial_2d.m is a routine for constructing basis functions of 
% complete ordinary polynomial of the degrees from one to five for the 
% two-dimensional case; see  "Envelope Condition Method with an Application 
% to Default Risk Models" by Cristina Arellano, Lilia Maliar, Serguei Maliar 
% and Viktor Tsyrennikov,  Journal of Economic Dynamics and Control (2016), 
% 69, 436-459 (henceforth, AMMT, 2016).   
% -------------------------------------------------------------------------

% -------------------------------------------------------------------------
% Inputs:    "s" is the data points on which the polynomial basis functions  
%               must be constructed; T-by-2;
%            "D" is the degree of the polynomial whose basis functions must 
%               be constructed; (can be 1,2,3,4 or 5)
%
% Output:    "x" is the matrix of basis functions of complete polynomial 
%               of the given degree 
% -------------------------------------------------------------------------
% Copyright © 2012-2016 by Lilia Maliar and Serguei Maliar. All rights 
% reserved. The code may be used, modified and redistributed under the 
% terms provided in the file "License_Agreement.txt".
% -------------------------------------------------------------------------

function x = Polynomial_2d(s,D)

T = size(s,1);  % s can be a scalar or vector; T-by-2 

p1 = s(:,1); p2=s(:,1).^2;p3=s(:,1).^3;p4=s(:,1).^4;p5=s(:,1).^5;
q1 = s(:,2); q2=s(:,2).^2;q3=s(:,2).^3;q4=s(:,2).^4;q5=s(:,2).^5;    

if D==1;
    x = [ones(T,1) p1 q1];
elseif D==2;
    x = [ones(T,1) p1 q1 p2 p1.*q1 q2];
elseif D==3;
    x = [ones(T,1) p1 q1 p2 p1.*q1 q2 p3 p2.*q1 p1.*q2 q3];
elseif D==4;
    x = [ones(T,1) p1 q1 p2 p1.*q1 q2 p3 p2.*q1 p1.*q2 q3 p4 p3.*q1 p2.*q2 p1.*q3 q4];
elseif D==5;
    x = [ones(T,1) p1 q1 p2 p1.*q1 q2 p3 p2.*q1 p1.*q2 q3 p4 p3.*q1 p2.*q2 p1.*q3 q4 p5 p4.*q1 p3.*q2 p2.*q3 p1.*q4 q5];
end    
