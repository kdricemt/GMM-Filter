%% Test GMM_DO
% Paper:
% Sondergaard, Thomas, and Pierre F. J. Lermusiaux. “Data Assimilation
% with Gaussian Mixture Models Using the Dynamically Orthogonal Field 
% Equations. Part II: Applications.” Monthly Weather Review (2012): 121011101334009.

clear all; close all;
rng(1) 

%% Double Well Diffusion Experiment
% dimension
n = 1;  % dimension of state vector
p = 1;  % dimension of observation vector

N = 1000;  % number of data realization
M = 2;     % gaussian mixture model complexity

t_end  = 40;
dt     = 1e-3;
t_vec  = 0:dt:t_end;
T      = length(t_vec);    % number of timesteps

% Forecast Parameters
pi_true = zeros(1,M);
mu_true = zeros(M,n);
Sigma_true = zeros(n,n,M);

pi_true(1) = 0.5;
mu_true(1,:) = -0.98;
Sigma_true(:,:,1) = 0.111;

pi_true(2) = 0.5;
mu_true(2,:) = 0.98;
Sigma_true(:,:,2) = 0.111;

gm_true = gmdistribution(mu_true,Sigma_true,pi_true);

% genrate realization data
phi_f = zeros(N,n); % realizations residing in stochastic subspace (forcast)

phi_idx = 1;

for j = 1:M
    nData = pi_true(j) * N;
    phi_f(phi_idx:phi_idx+nData-1,:) = mvnrnd(mu_true(j,:),Sigma_true(:,:,j),nData);
    phi_idx = phi_idx + nData;
end

% propagate the true state of the ball
kappa = 0.4;
r = 100;
dt_small = dt/r;
x_t = zeros(1,T);
x_t(1) = 1;

for t = 2:T
    dw = sqrt(dt_small) * randn(1,r);
    x_t(t) = x_t(t-1) + 4*(x_t(t-1) - x_t(t-1)^3) * dt + kappa * sum(dw);
end

% Plot x
figure(1)
hold on;
plot(t_vec,x_t,'k-');
ylim([-1.5 1.5]);

figure(2)
x = -1:0.01:1;
y = zeros(1,length(x));
for i = 1:length(x)
    y(i) = 4*(x(i) - x(i)^3);
end
plot(x,y);

