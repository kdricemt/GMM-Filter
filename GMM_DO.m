%% Test GMM_DO
% Paper:
% Sondergaard, Thomas, and Pierre F. J. Lermusiaux. “Data Assimilation
% with Gaussian Mixture Models Using the Dynamically Orthogonal Field 
% Equations. Part I: Theory and Scheme.” Monthly Weather Review (2012): 121011101334009.

clear all; close all;

%% Initial Settings
% dimension
n = 3;  % dimension of state vector
p = 2;  % dimension of observation vector
s = 2;  % dimension of stochastic subspace

N = 100;  % number of data realization
M = 2;    % gaussian mixture model complexity


x_bar_f = [1,2,3]';      % mean state vector 
Chi = [1 0; 
       0 1; 
       0 0];  % matrix of DO Modes (n x s)

% Forecast Parameters
pi_true = zeros(1,M);
mu_true = zeros(M,s);
Sigma_true = zeros(s,s,M);

pi_true(1) = 0.5;
mu_true(1,:) = [-10 -1];
Sigma_true(:,:,1) = [1 0;0 1];

pi_true(2) = 0.5;
mu_true(2,:) = [10 1];
Sigma_true(:,:,2) = [1 0;0 1];

gm_true = gmdistribution(mu_true,Sigma_true,pi_true);

% genrate realization data
phi_f = zeros(N,s); % realizations residing in stochastic subspace (forcast)

phi_idx = 1;

for j = 1:M
    nData = pi_true(j) * N;
    phi_f(phi_idx:phi_idx+nData-1,:) = mvnrnd(mu_true(j,:),Sigma_true(:,:,j),nData);
    phi_idx = phi_idx + nData;
end

% true state
x_t = x_bar_f + Chi*phi_f(1,:)';

% observation vector
H = [1 0 0;
     0 0 1];  % get noisy meas of first and third state

% error covariance matrx
sigma_obs = 5;
R = sigma_obs^2 * eye(p);

%% Run 1 Iteration of Estimation
% Step1: Run EM algorithm to obtain the prior mixture parameters
BIC = zeros(1,M);
GMModel = cell(1,M);

for mi = 1:M
    GMModel{mi} = fitgmdist(phi_f,mi);  % fitting using EM Algorithm
    BIC(:,mi) = GMModel{mi}.BIC;
end
[bestBIC,bestModel] = min(BIC);

pi_f    = GMModel{bestModel}.ComponentProportion;  % 1xM
mu_f    = GMModel{bestModel}.mu;                   % Mxs
Sigma_f = GMModel{bestModel}.Sigma;                % sxsxM

% True Observation
y =  H*x_t + mvnrnd(zeros(p,1),R,1)';  % px1

% Step2: Update
% 2-1 Calculate parameters
H_tilde = H*Chi;          % (p x n) x (n x s) = (p x s)
y_tilde = y - H*x_bar_f;   %  p x 1
K_tilde = zeros(s,p,M);   % (s x s) x (s x p) x (p x p) = (s x p)
for j = 1:M
    K_tilde(:,:,j) = Sigma_f(:,:,j) * H_tilde' / (H_tilde * Sigma_f(:,:,j) * H_tilde' + R);
end

% 2-2: Assimilate the Observations
pi_a = zeros(M,1);
mu_hat_a = zeros(size(mu_f));  % M x s

for j = 1:M
    mu_fj = mu_f(j,:)';
    mu_hat_a(j,:) = mu_fj + K_tilde(:,:,j) * (y_tilde - H_tilde * mu_fj);
    
    % posterior mixture
    mu_post = H_tilde*mu_fj;
    sig_post =  H_tilde * Sigma_f(:,:,j) * H_tilde' + R;
    pi_a(j) = pi_f(j) * mvnpdf(y_tilde, mu_post, sig_post);
end

pi_a = pi_a/sum(pi_a);

% 2-3. Update the DO Mean Field
sum_pi_mu = zeros(s,1);
for j = 1:M
    sum_pi_mu = sum_pi_mu + pi_a(j) .* mu_hat_a(j,:)';  % s x 1
end

x_bar_a = x_bar_f + Chi * sum_pi_mu;  % (3x2)x(2x1)

mu_a = zeros(size(mu_f));
Sigma_a = zeros(size(Sigma_f));
for j = 1:M
    mu_a(j,:) = mu_f(j,:) - sum_pi_mu';
    Sigma_a(:,:,j) = (eye(s) - K_tilde(:,:,j)*H_tilde)*Sigma_f(:,:,j);
end

% 2-4. Generate Posterior Sets of subspace ensemble realizations
phi_a = zeros(N,s); % realizations residing in stochastic subspace (forcast)

phi_idx = 1;
for j = 1:M
    nData = round(pi_a(j) * N);
    if nData > 1
        phi_a(phi_idx:phi_idx+nData-1,:) = mvnrnd(mu_a(j,:),Sigma_a(:,:,j),nData);
    end
    phi_idx = phi_idx + nData;
end

x_a = x_bar_a + Chi*phi_a';  % (n x s) x (s x N) = (n x N)
x_f = x_bar_f + Chi*phi_f';  % (n x s) x (s x N) = (n x N)

x_a_mean = mean(x_a,2);
x_f_mean = mean(x_f,2);

%% Post Process
postprocess = 1;

if postprocess
    close all;
    % PDF
    figure(1)
    gmPDF = @(x,y) arrayfun(@(x0,y0) pdf(gm_true,[x0 y0]),x,y);
    fsurf(gmPDF,[-15 15])

    % Realization 
    figure(2)
    hold on; 
    scatter(phi_f(:,1),phi_f(:,2),30,'.');
    
    % GMM Model Fitting 
    figure(3)
    hold on;
    sgtitle('Fitting of GMM to Ensemble Realization');
    for j = 1:M
        subplot(M,1,j);
        scatter(phi_f(:,1),phi_f(:,2),30,'r','.'); 
        gmPDF = @(x,y) arrayfun(@(x0,y0) pdf(GMModel{j},[x0 y0]),x,y);
        fcontour(gmPDF,[-15 15],'MeshDensity',100)
        ylim([-4 4]);
        xlim([-15 15]);
    end
    
    % Posterior Set of realizations
    figure(4)
    title('Posterior GMM (Red: Posterior  Black: Prior)');
    hold on; 
    scatter(phi_f(:,1),phi_f(:,2),30,'k','.'); 
    scatter(phi_a(:,1),phi_a(:,2),30,'r','.'); 
    ylim([-4 4]);
    xlim([-15 15]);
    
    % Solution
    figure(5)
    hold on; 
    plot(1:3, x_t, 'go-');
    plot(1:3, x_a_mean, 'ro-');
    plot(1:3, x_f_mean, 'bo-');
    errorbar([1],[y(1)],[sigma_obs], 'ko');
    errorbar([3],[y(2)],[sigma_obs], 'ko');
    legend({'True', 'Posterior Mean', 'Prior Mean', 'Measurement'}, 'Location','southeast');
    
end
