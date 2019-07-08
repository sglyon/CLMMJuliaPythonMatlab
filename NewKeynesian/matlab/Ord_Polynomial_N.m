% Ord_Polynomial_N.m is a routine that constructs the basis functions of 
% complete ordinary polynomial of the degrees from one to five for the 
% multi-dimensional case; see "Numerically Stable and Accurate Stochastic 
% Simulation Approaches for Solving Dynamic Economic Models" by Kenneth L. 
% Judd, Lilia Maliar and Serguei Maliar, (2011), Quantitative Economics 2/2, 
% 173–210 (henceforth, JMM, 2011).
%
% This version: July 14, 2011. First version: August 27, 2009.
% -------------------------------------------------------------------------
% Inputs:  "z" is the data points on which the polynomial basis functions  
%               must be constructed; n_rows-by-dimen; 
%          "D" is the degree of the polynomial whose basis functions must 
%               be constructed; (can be 1,2,3,4 or 5)
%
% Output:  "basis_fs" is the matrix of basis functions of a complete 
%               polynomial of the given degree 
% -------------------------------------------------------------------------
% Copyright © 2011 by Lilia Maliar and Serguei Maliar. All rights reserved. 
% The code may be used, modified and redistributed under the terms provided 
% in the file "License_Agreement.pdf".
% -------------------------------------------------------------------------

function basis_fs = Ord_Polynomial_N(z,D)


% A polynomial is given by the sum of polynomial basis functions, phi(i),
% multiplied by the coefficients; see condition (13) in JMM (2011). By 
% convention, the first basis function is one. 

[n_rows,dimen] = size(z); % Compute the number of rows, n_rows, and the  
                          % number of variables (columns), dimen, in the    
                          % data z on which the polynomial basis functions   
                          % must be constructed

    % 1. The matrix of the basis functions of the first-degree polynomial 
    % (the default option)
    % -------------------------------------------------------------------
    basis_fs = [ones(n_rows,1) z];  % The matrix includes a column of ones
                                    % (the first basis function is one for
                                    % n_rows points) and linear basis
                                    % functions
    i = dimen+1; % Index number of a polynomial basis function; the first  
                 % basis function (equal to one) and linear basis functions 
                 % are indexed from 1 to dimen+1, and subsequent polynomial 
                 % basis functions will be indexed from dimen+2 and on
    
    % 2. The matrix of the basis functions of the second-degree polynomial 
    % --------------------------------------------------------------------   
    if D == 2 
 
% Version one (not vectorized): 
        for j1 = 1:dimen   
            for j2 = j1:dimen
                i = i+1;
                basis_fs(:,i) = z(:,j1).*z(:,j2);
            end
        end

% Version 2 (vectorized): Note that this version works only for a second-degree 
% polynomial in which all state variables take non-zero values
%        for r = 1:n_rows
%            basis_fs(r,2+dimen:1+dimen+dimen*(dimen+1)/2) = [nonzeros(tril(z(r,:)'*z(r,:)))']; 
            % Compute linear and quadratic polynomial basis functions for 
            % each row r; "tril" extracts a lower triangular part of z'z 
            % (note that matrix z'z is symmetric so that an upper triangular 
            % part is redundant); "nonzeros" forms a column vector by 
            % stacking the columns of the original matrix one after another 
            % and by eliminating zero terms
 %       end
 
    % 3. The matrix of the basis functions of the third-degree polynomial 
    % -------------------------------------------------------------------    
    elseif D == 3                

        for j1 = 1:dimen
            for j2 = j1:dimen
                i = i+1;
                basis_fs(:,i) = z(:,j1).*z(:,j2);
                for j3 = j2:dimen
                    i = i+1;
                    basis_fs(:,i) = z(:,j1).*z(:,j2).*z(:,j3);
                end
            end
        end

    % 4. The matrix of the basis functions of the fourth-degree polynomial 
    % -------------------------------------------------------------------    
    elseif D == 4    
        
        for j1 = 1:dimen
            for j2 = j1:dimen
                i = i+1;
                basis_fs(:,i) = z(:,j1).*z(:,j2);
                for j3 = j2:dimen
                    i = i+1;
                    basis_fs(:,i) = z(:,j1).*z(:,j2).*z(:,j3);
                    for j4 = j3:dimen
                        i = i+1;
                        basis_fs(:,i) = z(:,j1).*z(:,j2).*z(:,j3).*z(:,j4);
                    end
                end
            end
        end

    % 5. The matrix of the basis functions of the fifth-degree polynomial 
    % ------------------------------------------------------------------- 

    elseif D == 5         
        
        for j1 = 1:dimen
            for j2 = j1:dimen
                i = i+1;
                basis_fs(:,i) = z(:,j1).*z(:,j2);
                for j3 = j2:dimen
                    i = i+1;
                    basis_fs(:,i) = z(:,j1).*z(:,j2).*z(:,j3);
                    for j4 = j3:dimen
                        i = i+1;
                        basis_fs(:,i) = z(:,j1).*z(:,j2).*z(:,j3).*z(:,j4);
                        for j5 = j4:dimen
                            i = i+1;
                            basis_fs(:,i) = z(:,j1).*z(:,j2).*z(:,j3).*z(:,j4).*z(:,j5);
                        end
                    end
                end
            end
        end
    
    end                         % end of "if"/"elseif"   
