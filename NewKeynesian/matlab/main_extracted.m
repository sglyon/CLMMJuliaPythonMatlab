function [running_time,Residuals_mean,Residuals_max] = main_extracted(in)

piestar = in.piestar;
zlb = in.zlb;
Degree = in.Degree;

% zlb        = 1;           % Impose ZLB on nominal interest rate
gam        = 1;           % Utility-function parameter
betta      = 0.99;        % Discount factor
vartheta   = 2.09;        % Utility-function parameter
epsil      = 4.45;        % Parameter in the Dixit-Stiglitz aggregator
phi_y      = 0.07;        % Parameter of the Taylor rule
phi_pie    = 2.21;        % Parameter of the Taylor rule
mu         = 0.82;        % Parameter of the Taylor rule
theta      = 0.83;        % Share of non-reoptimizing firms (Calvo's pricing)
% piestar    = 1;           % Target (gross) inflation rate
Gbar       = 0.23;        % Steady-state share of government spending in output

rho_nua    = 0.95;        % See process (17) in CLMM (2017)
rho_nuL    = 0.25;        % See process (11) in CLMM (2017)
rho_nuR    = 0.0;         % See process (23) in CLMM (2017)
rho_nuu    = 0.92;        % See process (10) in CLMM (2017)
rho_nuB    = 0.0;         % See process (12) in CLMM (2017)
rho_nuG    = 0.95;        % See process (21) in CLMM (2017)

sigma_nua  = 0.0045;      % See process (17) in CLMM (2017)
sigma_nuL  = 0.0500;      % See process (11) in CLMM (2017)
sigma_nuR  = 0.0028;      % See process (23) in CLMM (2017)
sigma_nuu  = 0.0054;      % See process (10) in CLMM (2017)
sigma_nuB  = 0.0010;      % See process (12) in CLMM (2017)
sigma_nuG  = 0.0038;      % See process (21) in CLMM (2017)

Yn_ss     = exp(Gbar)^(gam/(vartheta+gam));
Y_ss      = Yn_ss;
pie_ss    = 1;
delta_ss  = 1;
L_ss      = Y_ss/delta_ss;
C_ss      = (1-Gbar)*Y_ss;
F_ss      = C_ss^(-gam)*Y_ss/(1-betta*theta*pie_ss^(epsil-1));
S_ss      = L_ss^vartheta*Y_ss/(1-betta*theta*pie_ss^epsil);
R_ss      = pie_ss/betta;
w_ss      = (L_ss^vartheta)*(C_ss^gam);

% Degree  = 3;    % Degree of polynomial approximation can be 1,...,5
grid_type = 2;  % Choose a grid type; grid_type = 1 corresponds to a uniformely
                % distribued random grid, and  grid_type = 2 corresponds to
                % a quasi Monte Carlo grid
grid_sizes(1:5,1) = [20 100 300 1000 2000];
                % The number of grid points m(1),...,m(5) that corresponds to
                % the polynomial approximations of degrees 1,...,5 containing
                % 9,45,165,495 and 1287 coefficients, respectively
m = grid_sizes(Degree);
                % m is the number if points in the grid that corresponds to
                % given degree of polynomial approximation

if grid_type == 1 % If the grid is random
    %random_grids = rand(2000,8);
    load random_grids;
    random_points = random_grids(1:m,:);
                    % Restrict grid to contain m points
    nuR0    = (-2*sigma_nuR+4*sigma_nuR*random_points(:,1))/sqrt(1-rho_nuR^2);
    nua0    = (-2*sigma_nua+4*sigma_nua*random_points(:,2))/sqrt(1-rho_nua^2);
    nuL0    = (-2*sigma_nuL+4*sigma_nuL*random_points(:,3))/sqrt(1-rho_nuL^2);
    nuu0    = (-2*sigma_nuu+4*sigma_nuu*random_points(:,4))/sqrt(1-rho_nuu^2);
    nuB0    = (-2*sigma_nuB+4*sigma_nuB*random_points(:,5))/sqrt(1-rho_nuB^2);
    nuG0    = (-2*sigma_nuG+4*sigma_nuG*random_points(:,6))/sqrt(1-rho_nuG^2);
        % Values of exogenous state variables are distributed uniformly
        % in the interval +/- std/sqrt(1-rho_nu^2); see CLMM (2017)

    R0      = 1+0.05*random_points(:,7);
    delta0  = 0.95+0.05*random_points(:,8);
        % Values of endogenous state variables are distributed uniformly
        % in the intervals [1 1.05] and [0.95 1], respectively; see CLMM (2017)
end

%%
% 3.2 Quasi-random (Sobol) grid

if grid_type == 2  % If the grid is Sobol,...

    % Sobol_grids = net(sobolset(8),m);
    %                   % Constructs a Sobol sequence with m 8-dimensional points
    load Sobol_grids;
    Sobol_points = Sobol_grids(1:m,:);

    nuR0    = (-2*sigma_nuR+4*(max(Sobol_points(:,1))-Sobol_points(:,1))/(max(Sobol_points(:,1))-min(Sobol_points(:,1)))*sigma_nuR)/sqrt(1-rho_nuR^2);
    nua0    = (-2*sigma_nua+4*(max(Sobol_points(:,2))-Sobol_points(:,2))/(max(Sobol_points(:,2))-min(Sobol_points(:,2)))*sigma_nua)/sqrt(1-rho_nua^2);
    nuL0    = (-2*sigma_nuL+4*(max(Sobol_points(:,3))-Sobol_points(:,3))/(max(Sobol_points(:,3))-min(Sobol_points(:,3)))*sigma_nuL)/sqrt(1-rho_nuL^2);
    nuu0    = (-2*sigma_nuu+4*(max(Sobol_points(:,4))-Sobol_points(:,4))/(max(Sobol_points(:,4))-min(Sobol_points(:,4)))*sigma_nuu)/sqrt(1-rho_nuu^2);
    nuB0    = (-2*sigma_nuB+4*(max(Sobol_points(:,5))-Sobol_points(:,5))/(max(Sobol_points(:,5))-min(Sobol_points(:,5)))*sigma_nuB)/sqrt(1-rho_nuB^2);
    nuG0    = (-2*sigma_nuG+4*(max(Sobol_points(:,6))-Sobol_points(:,6))/(max(Sobol_points(:,6))-min(Sobol_points(:,6)))*sigma_nuG)/sqrt(1-rho_nuG^2);
        % Values of exogenous state variables are in the interval +/- std/sqrt(1-rho^2);
        % see CLMM (2017)

    R0      = 1+0.05*(max(Sobol_points(:,7))-Sobol_points(:,7))/(max(Sobol_points(:,7))-min(Sobol_points(:,7)));
    delta0  = 0.95+0.05*(max(Sobol_points(:,8))-Sobol_points(:,8))/(max(Sobol_points(:,8))-min(Sobol_points(:,8)));
        % Values of endogenous state variables are in the intervals [1 1.05] and
        % [0.95 1], respectively
end

%%
% 3.3. Imposing ZLB for the grid construction

if zlb == 1; R0=max(R0,1); end
             % If ZLB is imposed, set R(t)=1 when ZLB binds

Grid    = [log(R0) log(delta0) nuR0 nua0 nuL0 nuu0 nuB0 nuG0];
             % Construct the matrix of grid points; m-by-dimensionality

%%
% Section 4: Constructing ordinary polynomial function using the function
% Ord_Polynomial_N.m
X0_Gs{1} = Ord_Polynomial_N(Grid, 1);
X0_Gs{Degree} = Ord_Polynomial_N(Grid, Degree);
                     % Construct the matrix of explanatory variables X0_G
                     % on the grid of state variables; the columns of X0_G
                     % are given by the basis functions of polynomial of
                     % degree "Degree"
npol = size(X0_Gs{Degree}, 2); % Number of coefficients in polynomial of degree
                     % "Degree"; it must be smaller than the number of grid
                     % points
%%
% Step 5:Numerical integration using the functions Monomials_1.m and Monomials_2.m

%%
% Step 5: Computing shocks in the future states in the grid points

N = 6;               % Total number of exogenous shocks
vcv = diag([sigma_nuR^2 sigma_nua^2 sigma_nuL^2 sigma_nuu^2 sigma_nuB^2 sigma_nuG^2]);
                     % Variance covariance matrix

% Compute the number of integration nodes, their values and weights
%------------------------------------------------------------------
[n_nodes,epsi_nodes,weight_nodes] = Monomials_1(N,vcv);
                     % Monomial integration rule with 2N nodes
%[n_nodes,epsi_nodes,weight_nodes] = Monomials_2(N,vcv);
                     % Monomial integration rule with 2N^2+1 nodes

nuR1(:,1:n_nodes) = (nuR0*ones(1,n_nodes)).*rho_nuR + ones(m,1)*epsi_nodes(:,1)';
nua1(:,1:n_nodes) = (nua0*ones(1,n_nodes)).*rho_nua + ones(m,1)*epsi_nodes(:,2)';
nuL1(:,1:n_nodes) = (nuL0*ones(1,n_nodes)).*rho_nuL + ones(m,1)*epsi_nodes(:,3)';
nuu1(:,1:n_nodes) = (nuu0*ones(1,n_nodes)).*rho_nuu + ones(m,1)*epsi_nodes(:,4)';
nuB1(:,1:n_nodes) = (nuB0*ones(1,n_nodes)).*rho_nuB + ones(m,1)*epsi_nodes(:,5)';
nuG1(:,1:n_nodes) = (nuG0*ones(1,n_nodes)).*rho_nuG + ones(m,1)*epsi_nodes(:,6)';
            % Compute future shocks in all grid points and all integration
            % nodes; the size of each of these matrices is m-by-n_nodes
%%
% Section 6: Initial guess for coefficients of the decision functions for
% the variables S and F and marginal utility MU. Done in main loop...
%%
% Step 7 : Allocate Memory

e = zeros(m,3);
% Allocate memory to integrals in the right side of 3 Euler equations

S0_old_G = ones(m,1);
F0_old_G = ones(m,1);
C0_old_G = ones(m,1);
% Allocate memory to S, F and C from the previous iteration (to check
% convergence)

S0_new_G = ones(m,1);
F0_new_G = ones(m,1);
C0_new_G = ones(m,1);
% Allocate memory to S, F, C from the current iteration (to check
% convergence)

damp     = 0.1;           % Damping parameter for (fixed-point) iteration on
                          % the coefficients of 3 decision functions (for
                          % S, F and C^(-gam))

% We will run the code for the degree the user passed in `deg`
% However, if we are running
if Degree > 1
    degrees = [1, Degree];
else
    degrees = [Degree];
end

for deg = degrees
    tic;
    diff = 1e+10;
    X0_G = X0_Gs{deg};
    it = 0;

    if deg <= 2
         vk       =  ones(size(X0_G, 2), 3)*1e-5;  % Initialize first all the coefficients
                                                     % at 1e-5
         vk(1,:)  = [S_ss F_ss C_ss.^(-gam)];        % Set the initial values of the constant
                                                     % terms in the decision rules for S,
                                                     % F and MU to values that give the
                                                     % deterministic steady state
    else
        % For degree > 2, initial guess for coefficients is given by the
        % result of the degree 2 computation
        vk = X0_G\e;
%         vk_old = vk;
%         vk = ones(npol, 3)*1e-5;
%         vk(1:size(vk_old, 1), :) = vk_old;
    end

    while diff > 1e-7         % The convergence criterion (which is unit free
                              % because diff is unit free)
        it = it + 1;

        % Current choices (at t)
        % ------------------------------
        S0 = X0_G*vk(:,1);              % Compute S(t) using vk
        F0 = X0_G*vk(:,2);              % Compute F(t) using vk
        C0 = (X0_G*vk(:,3)).^(-1/gam);  % Compute C(t) using vk

        pie0 = ((1-(1-theta)*(S0./F0).^(1-epsil))/theta).^(1/(epsil-1));
                   % Compute pie(t) from condition (28) in CLMM (2017)
        delta1 = ((1-theta)*((1-theta*pie0.^(epsil-1))/(1-theta)).^(epsil/(epsil-1))+theta*pie0.^epsil./delta0).^(-1);
                  % Compute delta(t) from condition (29) in CLMM (2017)
        Y0 = C0./(1-Gbar./exp(nuG0));
                   % Compute Y(t) from condition (32) in CLMM (2017)
        L0 = Y0./exp(nua0)./delta1;
                   % Compute L(t) from condition (31) in CLMM (2017)
        Yn0 = (exp(nua0).^(1+vartheta).*(1-Gbar./exp(nuG0)).^(-gam)./exp(nuL0)).^(1/(vartheta+gam));
                   %  Compute Yn(t) from condition (34) in CLMM (2017)
        R1 = piestar/betta*(R0*betta./piestar).^mu.*((pie0./piestar).^phi_pie .* (Y0./Yn0).^phi_y).^(1-mu).*exp(nuR0);    % Taylor rule
                   % Compute R(t) from conditions (33) in CLMM (2017)
        if zlb == 1; R1 = max(R1,1); end
                   % If ZLB is imposed, set R(t)=1 when ZLB binds

        % Future choices (at t+1)
        %--------------------------------
        delta1_dupl = delta1*ones(1,n_nodes);
        R1_dupl = R1*ones(1,n_nodes);
        % Duplicate "delta1" and "R1" n_nodes times to create a matrix with
        % n_nodes identical rows; m-by-n_nodes

        for u = 1:n_nodes

            X1 = Ord_Polynomial_N([log(R1) log(delta1) nuR1(:,u) nua1(:,u) nuL1(:,u) nuu1(:,u) nuB1(:,u) nuG1(:,u)],deg);
            % Form complete polynomial of degree "Degree" (at t+1) on future state
            % variables; n_nodes-by-npol

            S1(:,u) = X1*vk(:,1);             % Compute S(t+1) in all nodes using vk
            F1(:,u) = X1*vk(:,2);             % Compute F(t+1) in all nodes using vk
            C1(:,u) = (X1*vk(:,3)).^(-1/gam); % Compute C(t+1) in all nodes using vk

        end

        pie1 = ((1-(1-theta)*(S1./F1).^(1-epsil))/theta).^(1/(epsil-1));
                                          % Compute next-period pie using condition
                                          % (28) in CLMM (2017)


       % Evaluate conditional expectations in the Euler equations
       %---------------------------------------------------------
       e(:,1) = exp(nuu0).*exp(nuL0).*L0.^vartheta.*Y0./exp(nua0) + (betta*theta*pie1.^epsil.*S1)*weight_nodes;
       e(:,2) = exp(nuu0).*C0.^(-gam).*Y0 + (betta*theta*pie1.^(epsil-1).*F1)*weight_nodes;
       e(:,3) = betta*exp(nuB0)./exp(nuu0).*R1.*((exp(nuu1).*C1.^(-gam)./pie1)*weight_nodes);


       % Variables of the current iteration
       %-----------------------------------
       S0_new_G(:,1) = S0(:,1);
       F0_new_G(:,1) = F0(:,1);
       C0_new_G(:,1) = C0(:,1);

       % Compute and update the coefficients of the decision functions
       % -------------------------------------------------------------
       vk_hat_2d = X0_G\e;   % Compute the new coefficients of the decision
                             % functions using a backslash operator

       vk = damp*vk_hat_2d + (1-damp)*vk;
                             % Update the coefficients using damping

       % Evaluate the percentage (unit-free) difference between the values
       % on the grid from the previous and current iterations
       % -----------------------------------------------------------------
       diff = mean(mean(abs(1-S0_new_G./S0_old_G)))+mean(mean(abs(1-F0_new_G./F0_old_G)))...
           +mean(mean(abs(1-C0_new_G./C0_old_G)));

       if mod(it, 20) == 0
           fprintf('On iteration %d err is %e\n', it, diff);
       end
                       % The convergence criterion is adjusted to the damping
                       % parameters
       % Store the obtained values for S(t), F(t), C(t) on the grid to
       % be used on the subsequent iteration
       %-----------------------------------------------------------------------
       S0_old_G = S0_new_G;
       F0_old_G = F0_new_G;
       C0_old_G = C0_new_G;
    end   % while
end  % for over degree
%%
% Step 10: Finish counting time


running_time  = toc;

%%
% Step 11: Simulating a time-series solution

T = 10201; % The length of stochastic simulation
%epsi_test_NK = randn(T,8);
load epsi_test_NK;

% Initialize the values of 6 exogenous shocks and draw innovations
%-----------------------------------------------------------------
nuR = zeros(T,1); eps_nuR = epsi_test_NK(:,1)*sigma_nuR;
nua = zeros(T,1); eps_nua = epsi_test_NK(:,2)*sigma_nua;
nuL = zeros(T,1); eps_nuL = epsi_test_NK(:,3)*sigma_nuL;
nuu = zeros(T,1); eps_nuu = epsi_test_NK(:,4)*sigma_nuu;
nuB = zeros(T,1); eps_nuB = epsi_test_NK(:,5)*sigma_nuB;
nuG = zeros(T,1); eps_nuG = epsi_test_NK(:,6)*sigma_nuG;

% Generate the series for shocks
%-------------------------------
for t = 1:T-1
    nuR(t+1,1) = rho_nuR*nuR(t,1) + eps_nuR(t);
    nua(t+1,1) = rho_nua*nua(t,1) + eps_nua(t);
    nuL(t+1,1) = rho_nuL*nuL(t,1) + eps_nuL(t);
    nuu(t+1,1) = rho_nuu*nuu(t,1) + eps_nuu(t);
    nuB(t+1,1) = rho_nuB*nuB(t,1) + eps_nuB(t);
    nuG(t+1,1) = rho_nuG*nuG(t,1) + eps_nuG(t)  ;
end

% Initial values of two endogenous state variables
%-------------------------------------------------
R_initial      = 1; % Nominal interest rate R
delta_initial  = 1; % Price dispersion "delta"

% Simulate the model
%-------------------
[S, F, delta, C, Y, Yn, L, R, pie, w] = NK_simulation(vk,nuR,nua,nuL,nuu,nuB,nuG,R_initial,delta_initial,gam,vartheta,epsil,betta,phi_y,phi_pie,mu,theta,piestar,Gbar,zlb,Degree);

%%
% Step 12 :Compute unit free Euler equation residuals on simulated points

discard = 200; % The number of observations to discard

[Residuals_mean(1) Residuals_max(1) Residuals_max_E(1:9,1) Residuals] = NK_accuracy(nua,nuL,nuR,nuG,nuB,nuu,R,delta,L,Y,Yn,pie,S,F,C,rho_nua,rho_nuL,rho_nuR,rho_nuu,rho_nuB,rho_nuG,gam,vartheta,epsil,betta,phi_y,phi_pie,mu,theta,piestar,vcv,discard,vk,Gbar,zlb,Degree);

end  % function
