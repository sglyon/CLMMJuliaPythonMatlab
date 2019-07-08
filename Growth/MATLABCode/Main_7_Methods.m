function growth_main(Method)
% This MATLAB software solves a neoclassical stochastic growth model using
% seven alternative solution methods and compares their accuracy and cost: 
% envelope condition method iterating on value function (ECM-VF), conventional 
% value function iteration (VFI), endogenous grid method (EGM), policy 
% iteration using envelope condition (ECM-PI), conventional policy iteration 
% (ECM-PI), envelope condition method iteration on the derivative of value 
% function (ECM-DVF) and conventional Euler equation method (EE). For a 
% description of these solution methods, see the article "Envelope Condition 
% Method with an Application to Default Risk Models" by Cristina Arellano, 
% Lilia Maliar, Serguei Maliar and Viktor Tsyrennikov, Journal of Economic 
% Dynamics and Control (2016), 69, 436-459 (henceforth, AMMT, 2016).  
%
% First version: December 12, 2012 
% This version:  October 31, 2016
% ------------------------------------------------------------------------
% The software uses the following files: 
% ------------------------------------------------------------------------
% 1. "Main_7.m"             solves the model and computes the residuals by 
%                           using seven alternative solution methods
% 2. "VF_Bellman.m"         evaluates value function on the left side of
%                           the Bellman equation
% 3. "FOC_VFI.m"            computes new capital that satisfies the FOC 
% 4. "EGM_BC.m"             computes current capital that satisfies budget 
%                           constraint given next-period capital 
% 5. "Simulation_ECM.m"     simulates a time-series solution given the 
%                           constructed value function
% 6. "Residuals_ECM.m       computes the residuals in equilibrium conditions
%                           on a given set of points in the state space for
%                           a given numerical solution
% 7. "Polynomial_2d.m"      constructs the sets of basis functions for ordinary
%                           polynomials of the degrees from one to five, for 
%                           the model with 2 state variables
% 9."Polynomial_deriv_2d.m" constructs the derivatives of basis functions of 
%                           complete ordinary polynomial of the degrees from 
%                           one to five, for the model with 2 state variables
% 9. "GH_Quadrature.m"      constructs integration nodes and weights for the 
%                           Gauss-Hermite rules with the number of nodes in
%                           each dimension ranging from one to ten; borrowed 
%                           from Judd, Maliar and Maliar (QE, 2011)
% 10. "epsi_test.mat"       contains a fixed sequence of random numbers
% -------------------------------------------------------------------------
% Copyright � 2012-2016 by Lilia Maliar and Serguei Maliar. All rights 
% reserved. The code may be used, modified and redistributed under the 
% terms provided in the file "License_Agreement.txt".
% -------------------------------------------------------------------------

fprintf('\n\n\n\n\nBeginning execution with method %i\n', Method)

% Method = 1;   % Choose a solution method: "1", "2", "3", "4", "5", "6", "7"

% "1" - envelope condition method iterating on value function (ECM-VF)
% "2" - conventional value function interation (VFI)
% "3" - endogenous grid method (EGM)
% "4" - policy function iteration via envelope condition (ECM-PI) 
% "5" - conventional policy function iteration via FOC (PI)
% "6" - envelope condition method iterating on derivative of value function (ECM-DVF)
% "7" - conventional Euler equation method (EE)

% 1. Model's parameters
% ---------------------
gam     = 2.0;       % Utility-function parameter: the code is stable for 
                     % the range of [0.08, 8]
alpha   = 0.36;      % Capital share in output
beta    = 0.99;      % Discount factor
delta   = 0.02;      % Depreciation rate 
rho     = 0.95;      % Persistence of the log of the productivity level
sigma   = 0.01;      % Standard deviation of shocks to the log of the 
                     % productivity level
                     
% 2. Technology level
% -------------------
A       = (1/beta-(1-delta))/alpha;  
                     % Normalize steady state capital to one             
css     = A-delta;   % Steady state consumption 
yss     = A;         % Steady state output


% 3. Construct a tensor product grid for computing a solution
%------------------------------------------------------------
% Unidimensional grid for capital
kmin    = 0.9;          % min capital on the grid
kmax    = 1.1;          % max capital on the grid
n_gridk = 10;           % number of grid points for capital 
gridk   = linspace(kmin,kmax,n_gridk)'; 
                        % Unidimensional grid for capital of n_grid points

% Unidimensional grid for productivity                        
zmin    = 0.9;          % min productivity on the grid
zmax    = 1.1;          % max productivity on the grid
n_gridz = 10;           % number of grid points for productivity 
gridz   = linspace(zmin,zmax,n_gridz)'; 
                        % Unidimensional grid for productivity

% Two-dimensional tensor product grid 
grid = []; 
for i = 1:n_gridz;
    grid = [grid;[gridk ones(n_gridk,1)*gridz(i,1)]]; 
end

grid_EGM = grid;        % Grid for the EGM method  

n_grid   = n_gridz*n_gridk;  
                        % Number of points in the two-dimensional grid 

k0 = grid(:,1);         % Grid points for capital in the tensor product grid
z0 = grid(:,2);         % Grid points for productivity in the tensor product grid
    
% 4. Gauss Hermite quadrature
%----------------------------
Qn      = 5;         % Number of integration nodes in Gauss Hermite quadrature
nshocks = 1;         % Number of stochastic shocks is 1
vcv     = sigma^2;   % Variance covariance matrix
[n_nodes,epsi_nodes,weight_nodes] = GH_Quadrature(Qn,nshocks,vcv);
                     % Gauss Hermite quadrature: number of nodes, values of 
                     % nodes and their weights, respectively
z1 = z0.^rho*exp(epsi_nodes');  
            % Compute future shocks in all grid points and all integration
            % nodes; the size is n_grid-by-n_nodes; 


% 5. Constructing initial guess for value function on the grid 
% ------------------------------------------------------------
D  = 2;              % Initial guess for value function is constructed using
                     % ordinary polynomial of degree 2
                     
X0   = Polynomial_2d(grid,D);
                     % Construct the matrix of explanatory variables X0 
                     % on the grid; the columns of X0 are given by basis 
                     % functions of polynomial of degree "D"        
                     
kdamp     = 1;       % Damping parameter for (fixed-point) iteration on 
                     % value function; a default value "1" means no damping
                     % and a smaller value in the interval (0,1) means that 
                     % value function is updated only partially from one
                     % iteration to another to enhance convergence
                     
vf_coef = zeros(6,1);% Initial guess for the coefficients of value function 

V = X0*vf_coef;      % Initialize the value function on the grid; this
                     % grid function will be used to check the convergence 

difv       = 1e+10;  % Initially, set the difference between the value 
                     % functions on two subsequent iterations to exceed 
                     % the convergence criterion

c0   =  A*z0.*k0.^alpha*(css/yss);        % Initial guess for consumption 
k1   =  k0*(1-delta)+A*z0.*k0.^alpha-c0;  % Initial guess for capital
                     % Our initial guess for consumption function is that 
                     % a constant fraction css/yss of the period output 
                     % goes to consumption and the rest goes to investment,
                     % where css/yss is calculated in the steady state
                      
while difv > 1e-6                          % Unit-free convergence criterion
    
        [V_new] = VF_Bellman(c0,k1,z1,gam,beta,n_nodes,weight_nodes,vf_coef,D);

        vf_coef_new = X0\V_new;            % New vector of coefficients 
        
        vf_coef = kdamp*vf_coef_new + (1-kdamp)*vf_coef;   
                                           % Update the coefficients
        difv = max(abs(1-V_new./V));       % Compute the difference between
                                           % new and old value functions

        V = V_new;                         % Store new value function

end

% 5. Main iterative cycle: constructing polynomial approximations of degrees 
% from 2 to 5 for value function 
% --------------------------------------------------------------------------

for D = 2:5;                            % For polynomial degrees from 2 to 5...   
        
    tic;
    Degree(D) = D;                      % Degree of polynomial approximation
    X0 = Polynomial_2d(grid,D);         % Ordinary polynomial of degree D 
    X0der = Polynomial_deriv_2d(grid,D);% Derivative of the polynomial
    vf_coef = X0\V;                     % Initial guess for value function
    K_coef  = X0\k1;                    % Initial guess for policy function
    difk =   inf;                       % Initial criterion of convergence 
    k_old = inf(size(grid,1),1);        % Initialize capital choices on the 
                                        % grid for checking convergence 
    opts = optimset('Display','none','Algorithm','trust-region-dogleg','MaxFunEvals',10000,'MaxIter',1000,'TolX',1e-10,'TolFun',1e-10);            
                                        % Options for the solver
    
    while difk > 1e-9;                  % Convergence criterion

        
        %  Method 1. Envelope condition method iterating on value function (ECM-VF)
        %==================================================================
        if Method==1        
        
            Vder0 = X0der*vf_coef;      % Compute derivative of value function
            u0 = Vder0./(1-delta+A*alpha*z0.*k0.^(alpha-1));
                                        % Computer marginal utility from
                                        % envelope condition
            c0 = u0.^(-1/gam);          % Find consumption   
            k1 = (1-delta)*k0+A*z0.*k0.^alpha-c0;  
                                        % Find capital from budget constraint
            [V_new] = VF_Bellman(c0,k1,z1,gam,beta,n_nodes,weight_nodes,vf_coef,D);
                                        % Recompute value function using 
                                        % Bellman equation
            vf_coef_new = X0\V_new;     % New coefficients for value function       
            vf_coef = kdamp*vf_coef_new + (1-kdamp)*vf_coef;   
                                        % Update the coefficients using 
                                        % damping
        %==================================================================

        
        % Method 2. Conventional value function iteration (VFI)
        %==================================================================
        elseif Method==2;            
            
            for i = 1:n_grid       % For each grid point 
                k1(i) = fsolve('FOC_VFI',k1(i),opts,k0(i),z0(i),A,alpha,gam,delta,beta,z1(i,:),n_nodes,weight_nodes,vf_coef,D);
                                   % Solve for capital that satisfies FOC
            end
            c0 = (1-delta)*k0+A*z0.*k0.^alpha-k1; 
                                   % Find consumption from budget constraint
            [V_new] = VF_Bellman(c0,k1,z1,gam,beta,n_nodes,weight_nodes,vf_coef,D);
                                   % Recompute value function using Bellman 
                                   % equation
            vf_coef_new = X0\V_new;
                                   % New coefficients for value function       
            vf_coef = kdamp*vf_coef_new + (1-kdamp)*vf_coef;   
                                   % Update the coefficients using damping
        %==================================================================
                                                             
                                   
        % Method 3. Endogenous grid method (EGM)
        %==================================================================                                   
        elseif Method==3;         
            
            k1 = grid_EGM(:,1); % Grid points for next-period capital 
                                % (fixing endogenous grid)
            for j = 1:n_nodes   
                
                X1_EGM    = Polynomial_2d([k1 z1(:,j)],D);                
                V1_EGM(:,j) = X1_EGM*vf_coef; % Compute value function in 
                                              % the integration nodes    

                Xder1_EGM = Polynomial_deriv_2d([k1 z1(:,j)],D);
                Vder1_EGM(:,j) = Xder1_EGM*vf_coef; % Compute derivative 
                                                    % of value function 
                                                    % in the integration nodes    
            end
            c0 = (beta*Vder1_EGM*weight_nodes).^(-1/gam);
            
            for i = 1:n_grid       % For each grid point 
                k0(i) = fsolve('EGM_BC',k0(i),opts,k1(i),z0(i),c0(i),A,alpha,delta);
                                   % Solve for current capital that satisfies 
                                   % budget constraint given next-period capital
            end
            

            if gam==1
                V_new = log(c0)+beta*V1_EGM*weight_nodes; 
                                   % Bellman equation if gam=1        
            else
                V_new = (c0.^(1-gam)-1)/(1-gam)+beta*V1_EGM*weight_nodes;
            end                    % Bellman equation otherwise
                     
            grid(:,1) = k0;        % Grid points for current capital 
            
            X0 = Polynomial_2d(grid,D);   % Construct polynomial on 
                                          % current state variables
            
            vf_coef_new = X0\V_new;
                                   % New coefficients for value function       
            vf_coef = kdamp*vf_coef_new + (1-kdamp)*vf_coef;   
                                   % Update the coefficients using damping
        %==================================================================
                                                            
                                                           
        % Method 4. Policy function iteration via envelope condition (ECM-PI)
        %==================================================================                                                                     
        elseif Method==4 
       
            k1 = X0*K_coef;      % Compute capital on the grid using the
                                 % given policy function
            c0 = (1-delta)*k0+A*z0.*k0.^alpha-k1;    
                                 % Compute consumption
           
            % Solve for value function from Bellman equation                      
            difv = 1e10;         % Initially, convergence criterion is not
                                 % satisfied
            while difv > 1e-6    % Unit-free convergence criterion       
                [V_new] = VF_Bellman(c0,k1,z1,gam,beta,n_nodes,weight_nodes,vf_coef,D);
                                 % Recompute value function using Bellman equation
                vf_coef_new = X0\V_new;         % New vector of coefficients 
                vf_coef = kdamp*vf_coef_new + (1-kdamp)*vf_coef;   
                                                % Update the coefficients
                difv = max(abs(1-V_new./V));    % Compute the difference between
                                                % new and old value functions
                V = V_new;                      % Store new value function
            end
            
            Vder0 = X0der*vf_coef;      % Compute derivative of value function
            u0 = Vder0./(1-delta+A*alpha*z0.*k0.^(alpha-1));
                                        % Compute marginal utility from
                                        % envelope condition
            c0 = u0.^(-1/gam);          % Find consumption   
            k1_new = (1-delta)*k0+A*z0.*k0.^alpha-c0;  
                                        % Find capital from budget constraint
            K_coef_new = X0\k1_new;     % New coefficients for policy function 
            K_coef = kdamp*K_coef_new + (1-kdamp)*K_coef;               
                                        % Update the coefficients using 
                                        % damping
        %==================================================================
  
        
        
        % Method 5. Conventional policy function iteration via FOC (PI)
        %==================================================================                                                               
        elseif Method==5  
                          
            k1 = X0*K_coef;      % Compute capital on the grid using the
                                 % given policy function
            c0 = (1-delta)*k0+A*z0.*k0.^alpha-k1;    
                                 % Compute consumption
           
            % Solve for value function from Bellman equation                      
            difv = 1e10;         % Initially, convergence criterion is not
                                 % satisfied
            while difv > 1e-6    % Unit-free convergence criterion       
                [V_new] = VF_Bellman(c0,k1,z1,gam,beta,n_nodes,weight_nodes,vf_coef,D);
                                 % Recompute value function using Bellman equation
                vf_coef_new = X0\V_new;         % New vector of coefficients 
                vf_coef = kdamp*vf_coef_new + (1-kdamp)*vf_coef;   
                                                % Update the coefficients
                difv = max(abs(1-V_new./V));    % Compute the difference between
                                                % new and old value functions
                V = V_new;                      % Store new value function
            end
          
            % Recompute the capital policy function using FOC
            for i = 1:n_grid     % For each grid point 
                k1_new(i,1) = fsolve('FOC_VFI',k1(i),opts,k0(i),z0(i),A,alpha,gam,delta,beta,z1(i,:),n_nodes,weight_nodes,vf_coef,D);
                                 % Solve for capital that satisfies FOC
            end
            K_coef_new = X0\k1_new;     % New coefficients for policy function 
            K_coef = kdamp*K_coef_new + (1-kdamp)*K_coef;               
                                        % Update the coefficients using 
                                        % damping
        %==================================================================
        
        
        
        % Method 6. Envelope condition method iterating on derivative of 
        % value function (ECM-DVF)
        %==================================================================
        elseif Method==6  
        
            Vder0 = X0der*vf_coef;      % Compute derivative of value function
            u0 = Vder0./(1-delta+A*alpha*z0.*k0.^(alpha-1));
                                        % Compute marginal utility from
                                        % envelope condition
            c0 = u0.^(-1/gam);          % Find consumption   
            k1 = (1-delta)*k0+A*z0.*k0.^alpha-c0;  
                                        % Find capital from budget constraint
                                        
            for j = 1:n_nodes           
                X1der = Polynomial_deriv_2d([k1 z1(:,j)],D);
                Vder1(:,j) = X1der*vf_coef; % Compute derivative of value
                                            % function in the integration nodes
            end
            
            Vder0_new = (1-delta+A*alpha*z0.*k0.^(alpha-1)).*(beta*Vder1*weight_nodes);
                                        % Recompute derivative of value 
                                        % function in the grid points
            warning('off')              % Some polynomial terms are zero for
                                        % the derivative and system is 
                                        % underdetermined. Least square 
                                        % problem is still correctly 
                                        % processed by a truncated QR method
                                        % but the system produces a warning
            vf_coef_new = X0der\Vder0_new;
                                   % New coefficients for value function       
            vf_coef = kdamp*vf_coef_new + (1-kdamp)*vf_coef;   
                                   % Update the coefficients using damping
        %==================================================================
        
        
        
        % Method 7. Conventional Euler equation method (EE)
        %==================================================================               
        elseif Method==7  
            
            k1 = X0*K_coef;      % Compute capital on the grid using the
                                 % given policy function
            for j = 1:n_nodes    
                X1 = Polynomial_2d([k1 z1(:,j)],D);            
                k2(:,j) = X1*K_coef; % Compute capital in the integration nodes
            end
            
            k1_dupl = k1*ones(1,n_nodes);
                     % Duplicate k1 n_nodes times to create a matrix with 
                     % n_nodes identical columns;   
            c1  = (1-delta)*k1_dupl+A*z1.*k1_dupl.^alpha-k2;
                     % Compute consumption in all integration nodes
            c0  =  (beta*(c1.^(-gam).*(1-delta+alpha*A*k1_dupl.^(alpha-1).*z1))*weight_nodes).^(-1/gam);
                     % Compute current consumption using Euler equation                    
            k1_new =(1-delta)*k0+A*z0.*k0.^alpha-c0;    
                     % Compute new capital on the grid
            K_coef_new = X0\k1_new;     
                    % New coefficients for policy function 
            K_coef = kdamp*K_coef_new + (1-kdamp)*K_coef;               
                    % Update the coefficients using damping      
        %==================================================================
                          
        end               
        
        % Checking convergence 
        % --------------------
        if Method==3 
            difk = max(abs(1-k0./k_old));    
                        % For EGM, we check convergence of current capital
            k_old = k0;        
        else         
            difk = max(abs(1-k1./k_old));
                        % For other methods, we check convergence of next 
                        % period capital
            k_old = k1;
        end
    end
    
    % After the solution is computed by any method, we construct the value 
    % value function for the constructed policy rules
    
    difv = 1e10;           % Initially, convergence criterion is not
                           % satisfied
    while difv > 1e-10     % Unit-free convergence criterion
        
        [V_new] = VF_Bellman(c0,k1,z1,gam,beta,n_nodes,weight_nodes,vf_coef,D);
        vf_coef_new = X0\V_new;         % New vector of coefficients 
        vf_coef = kdamp*vf_coef_new + (1-kdamp)*vf_coef;   
                                        % Update the coefficients
        difv = max(abs(1-V_new./V));    % Compute the difference between
                                        % new and old value functions
        V = V_new;                      % Store new value function
    end  

    V = V_new;                         % Update value function to be 
                                       % used as an initial guess for a
                                       % higher degree polynomials
    CPU(D) = toc;                      % Store running time
    VK(1:1+D+D*(D+1)/2,D) = vf_coef;   % Store the solution coefficients 
       
end

% Evaluate residuals in the model's equations on a simulated path
% ---------------------------------------------------------------
fprintf(1,'Method = %i:\n',Method);
fprintf(1,'ACCURACY EVALUATION AND RUNNING TIME:\n\n');
for D = 2:5 % For polynomial degrees from 2 to 5... 
    [k,z] = Simulation_ECM(A,alpha,gam,delta,sigma,rho,VK(1:1+D+D*(D+1)/2,D),D);
            % Simulate the solution under a sequence of 10,200 shocks stored
            % "epsi_test.mat" and discard the first 200 entries to eliminate
            % the effect of initial conditions
    [Mean_Residuals(D),Max_Residuals(D)] = Residuals_ECM(k,z,A,alpha,gam,delta,sigma,rho,beta,VK(1:1+D+D*(D+1)/2,D),D);
            % Compute residuals in the model's equations in each point of the 
            % simulated path and compute the mean and maximum residuals in
            % the model's equations
    fprintf(1,'Polynomial of degree = %i:\nRunning time = %.2f, Mean residuals = %.2f, Max residuals = %.2f\n\n',Degree(D),CPU(D),Mean_Residuals(D),Max_Residuals(D));
            % Display the results
end

end  % function