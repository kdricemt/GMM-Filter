% GMM EKF
rng(1)

%%  Ex1: Linear System
% x(n+1) = Ax(n) + Bu(n) + W(n)
% y(n) = Cx(n) + V(n)
% u(n) = -Ux(n)

A = [1.0 0.9;
     -0.5 1.2];
B = eye(2);
C = eye(2);
K_V = 0.1 * eye(2);
U = [0.5 0;
     0 0.7];

n = 2;  % state dimension
m = 2;  % observation dimension
T = 10; % number of timesteps
M = 3;  % complexity of GMM 

% True Noise Model (W) ------------------------------------------------
M_W = 3;
W_w_true = zeros(1,M_W);
W_mu_true = zeros(M_W,n);
W_Sigma_true = zeros(n,n,M_W);

W_w_true(1) = 0.4;
W_mu_true(1,:) = [-0.3 -0.3];
W_Sigma_true(:,:,1) = 0.02*eye(2);

W_w_true(2) = 0.3;
W_mu_true(2,:) = [0 0];
W_Sigma_true(:,:,2) = 0.02*eye(2);

W_w_true(3) = 0.3;
W_mu_true(3,:) = [0.3 0.3];
W_Sigma_true(:,:,3) = 0.02*eye(2);

W_wcum_true = [W_w_true(1), W_w_true(1)+W_w_true(2), W_w_true(1)+W_w_true(2)+W_w_true(3)];
W_gm_true = gmdistribution(W_mu_true,W_Sigma_true,W_w_true);

% Initial X Distribution  -------------------------------------------------
x0_w_true = zeros(1,M);
x0_mu_true = zeros(M,n);
x0_Sigma_true = zeros(n,n,M);

x0_w_true(1) = 0.3;
x0_mu_true(1,:) = [-3 -1];
x0_Sigma_true(:,:,1) = 0.2*eye(2);

x0_w_true(2) = 0.3;
x0_mu_true(2,:) = [3  1];
x0_Sigma_true(:,:,2) = 0.5*eye(2);

x0_w_true(3) = 0.4;
x0_mu_true(3,:) = [0  0];
x0_Sigma_true(:,:,3) = 0.2*eye(2);

x0_wcum_true = [x0_w_true(1), x0_w_true(1)+x0_w_true(2), x0_w_true(1)+x0_w_true(2)+x0_w_true(3)];
x0_gm_true = gmdistribution(x0_mu_true,x0_Sigma_true,x0_w_true);

% true state initialization ----------------------------------------------------
x_true = zeros(n,T);

% Estimate initial GMM Param using EM Algorithm
w     = zeros(M,T);

% initial state
% sample from true GMM
r = rand(1);
sample_j = min(find(x0_wcum_true > r));
x0 = mvnrnd(x0_mu_true(sample_j,:),x0_Sigma_true(:,:,sample_j),1);
x_true(:,1) = x0;
    
%% Initial GMM Estimation using EM algorithm
% Generate 100 samples
% genrate realization data
N = 100;
phi = zeros(N,n); 
phi_idx = 1;

for j = 1:M
    nData = x0_w_true(j) * N;
    phi(phi_idx:phi_idx+nData-1,:) = mvnrnd(x0_mu_true(j,:),x0_Sigma_true(:,:,j),nData);
    phi_idx = phi_idx + nData;
end

% Run EM Algorithm for the realization -------------------------
M_max = 4;
BIC = zeros(1,M_max);
GMModel = cell(1,M_max);

for j = 1:M_max
    GMModel{j} = fitgmdist(phi,j);  % fitting using EM Algorithm
    BIC(:,j) = GMModel{j}.BIC;
end
[bestBIC,bestM] = min(BIC);

w0     = GMModel{bestM}.ComponentProportion;  % 1xM
mu0    = GMModel{bestM}.mu;                   % Mxn
Sigma0 = GMModel{bestM}.Sigma;                % nxnxM

M = bestM;  % chosse the one with best BIC
 
xhat0 = zeros(1,n);
for j = 1:M
    xhat0 = xhat0 + w0(j) * mu0(j,:);
end

% param
x_bar = zeros(n,M,T);        % for each GMM
x_hat = zeros(size(x_bar));  % for each GMM
X_hat = zeros(n,T); 
x_diff = zeros(n,T);
S     = zeros(n,n,M,T);  % cov of x_bar
Sigma = zeros(n,n,M,T);  % cov of x_hat
y     = zeros(n,T);
u     = zeros(n,T);

x_bar(:,:,1)   = mu0';
x_hat(:,:,1)   = mu0';
S(:,:,:,1)     = Sigma0;
Sigma(:,:,:,1) = Sigma0;

for j = 1:M
   X_hat(:,1) = X_hat(:,1) + w0(j).*x_hat(:,j,1);
end

u(:,1) = -U * X_hat(:,1);
x_diff(:,1) = x_true(:,1) - x_hat(:,1);

%% Kalman Filtering
for t = 2:T
    % sample from true GMM
    r = rand(1);
    sample_j = min(find(W_wcum_true > r));
    W =  mvnrnd(W_mu_true(sample_j,:),W_Sigma_true(:,:,sample_j),1);
    
    % propagate true orbit
    x_true(:,t) = A * x_true(:,t-1) + B*u(:,t-1) + W';
    
    % generate observation
    y(:,t) = C*x_true(:,t) + mvnrnd(zeros(n,1),K_V,1)';
    
    w_tj = zeros(1,M);
    
    for j = 1:M
        % Time Update ------------------------------
        x_bar(:,j,t) = A * x_hat(:,j,t-1);
        
        % Meas Update --------------------------------
        S_jt = A * Sigma(:,:,j,t-1)*A' + K_V;
        K = S_jt * C' / (C * S_jt * C' + K_V);
        Sigma_jt = (eye(n) - K*C) * S_jt;
        
        % preserve
        S(:,:,j,t) = S_jt;
        Sigma(:,:,j,t) = Sigma_jt;
        
        x_hat(:,j,t) = x_bar(:,j,t) + K * (y(:,t) - C*x_bar(:,j,t));
        
        mu_post  = C * x_bar(:,j,t);
        sig_post = C * S_jt * C' + K_V;
        
        w_tj(j) = w(j,t-1) * mvnpdf(y(:,t),mu_post,sig_post);
    end
    
    w_tj = w_tj/sum(w_tj);
    w(:,t) = w_tj;
    
    X_hat(:,t) = 0;
    for j = 1:M
       X_hat(:,t) = X_hat(:,t) + w_tj(j).*x_hat(:,j,t);
    end
    
    u(:,t) = -U * X_hat(:,t);
    x_diff(:,t) = x_true(:,t) - x_hat(:,t);
end

%% Postprocessing
postprocess = 1;

if postprocess
    close all;
    % W PDF
    figure(1)
    title('Noise Model');
    gmPDF = @(x,y) arrayfun(@(x0,y0) pdf(W_gm_true,[x0 y0]),x,y);
    fsurf(gmPDF,[-2 2])
    view([-134 18]);
    
    % PDF
    figure(1)
    gmPDF = @(x,y) arrayfun(@(x0,y0) pdf(W_gm_true,[x0 y0]),x,y);
    fsurf(gmPDF,[-2 2])
    view([-134 18]);
end