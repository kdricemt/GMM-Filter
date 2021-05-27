% GMM EKF
% Debdipta Goswami and Derek A. Paley "Non-Gaussian Estimation and Dynamic 
% Output Feedback Using the Gaussian Mixture Kalman Filter", JGCD, 2021
clear all; close all;
rng(1)

%%  Ex1: Linear System
% x(n+1) = Ax(n) + Bu(n) + W(n)
% y(n) = Cx(n) + V(n)
% u(n) = -Ux(n)
test_case = 2;

n     = 2;  % state dimension
m     = 2;  % observation dimension
T     = 15; % number of timesteps
M     = 3;  % complexity of GMM (of init state estimate)
M_max = 4;  % maximum complexity of GMM
sigma_threshold = 10.0;

switch test_case
    case 1
        dynfun = @(x) dynfun1(x);
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
        
    case 2
        
        dynfun = @(x) dynfun2(x);
        % Initial X Distribution  -------------------------------------------------
        x0_w_true = zeros(1,M);
        x0_mu_true = zeros(M,n);
        x0_Sigma_true = zeros(n,n,M);

        x0_w_true(1) = 0.3;
        x0_mu_true(1,:) = [-1 -1];
        x0_Sigma_true(:,:,1) = 0.2*eye(2);

        x0_w_true(2) = 0.3;
        x0_mu_true(2,:) = [1  1];
        x0_Sigma_true(:,:,2) = 0.5*eye(2);

        x0_w_true(3) = 0.4;
        x0_mu_true(3,:) = [0  0];
        x0_Sigma_true(:,:,3) = 0.2*eye(2);        
end

C = eye(2);
K_V = 0.1 * eye(2);

% True Noise Model (W) ------------------------------------------------
M_W = 3;
W_w_true = zeros(1,M_W);
W_mu_true = zeros(M_W,n);
W_Sigma_true = zeros(n,n,M_W);

W_w_true(1) = 0.3;
W_mu_true(1,:) = [-0.3 -0.3];
W_Sigma_true(:,:,1) = 0.02*eye(2);

W_w_true(2) = 0.4;
W_mu_true(2,:) = [0 0];
W_Sigma_true(:,:,2) = 0.02*eye(2);

W_w_true(3) = 0.3;
W_mu_true(3,:) = [0.3 0.3];
W_Sigma_true(:,:,3) = 0.02*eye(2);

W_gm_true = gmdistribution(W_mu_true,W_Sigma_true,W_w_true);

% Estimate second moment of W 
N = 500;
phi_W = zeros(N,n); 
phi_idx = 1;
for j = 1:M
    nData = W_w_true(j) * N;
    phi_W(phi_idx:phi_idx+nData-1,:) = mvnrnd(W_mu_true(j,:),W_Sigma_true(:,:,j),nData);
    phi_idx = phi_idx + nData;
end
GModel_W = fitgmdist(phi_W,1);
K_W = GModel_W.Sigma;


x0_gm_true = gmdistribution(x0_mu_true,x0_Sigma_true,x0_w_true);

% true state initialization ----------------------------------------------------
x_true = zeros(n,T);

% initial state
% sample from true GMM
x0 = genSampleGMM(1,x0_w_true,x0_mu_true,x0_Sigma_true);
x_true(:,1) = x0;
    
%% Initial GMM Estimation using EM algorithm
N = 200;
phi_x0 = genSampleGMM(N,x0_w_true,x0_mu_true,x0_Sigma_true);
[w0,mu0,Sigma0,M] = sampleFitGMM(phi_x0,M_max);

% param ---------------------------------------------------
x_bar = zeros(n,M_max,T);        % for each GMM
x_hat = zeros(size(x_bar));  % for each GMM
X_hat = zeros(n,T); 
x_diff = zeros(n,T);
S     = zeros(n,n,M_max,T);      % cov of x_bar
Sigma = zeros(n,n,M_max,T);      % cov of x_hat 
w_bar  = zeros(M_max,T);          % weight of GMM (apriori)
w_hat  = zeros(M_max,T);          % weight of GMM (posterior)
y     = zeros(n,T);
u     = zeros(n,T);
x_diff_norm = zeros(1,T);
M_size = zeros(1,T);
GMMModel = cell(1,T);

% assign t=0 param --------------------------------------
x_bar(:,1:M,1)   = mu0';
x_hat(:,1:M,1)   = mu0';
S(:,:,1:M,1)     = Sigma0;
Sigma(:,:,1:M,1) = Sigma0;
w_bar(1:M,1)      = w0;
w_hat(1:M,1)      = w0;

for j = 1:M
   X_hat(:,1) = X_hat(:,1) + w0(j).*x_hat(:,j,1);
end

x_diff(:,1) = x_true(:,1) - x_hat(:,1);
x_diff_norm(1) = norm(x_diff(:,1));

M_size(1) = M;
GMMModel{1} = gmdistribution(mu0,Sigma0,w0);

%% Kalman Filtering
for t = 2:T 
    % True state & Observation Simulation --------------------------------
    % sample process noise from true GMM
    W = genSampleGMM(1,W_w_true,W_mu_true,W_Sigma_true);
    
    % propagate true orbit
    [x_true(:,t),~,u(:,t-1)] = dynfun(x_true(:,t-1));
    
    % generate observation
    y(:,t) = C*x_true(:,t) + mvnrnd(zeros(n,1),K_V,1)';
    
    % Time Update --------------------------------------------------------
    for j = 1:M
        [x_bar(:,j,t),AA,~] = dynfun(x_hat(:,j,t-1));
        S(:,:,j,t)   = AA * Sigma(:,:,j,t-1) * AA' + K_W;
        w_bar(j,t)   = w_hat(j,t-1);
    end
        
    % 2. Judge if resample Data -------------------------------------------
    % resample when apriori estimate covariance is over threshold
    if ~isempty(find(S(:,:,:,t) > sigma_threshold))
        disp('Resample GMM')
        % Generate Samples from previous xhat
        N = 500;
        phi_xhat = genSampleGMM(N,w_hat(1:M,t-1),x_hat(:,1:M,t-1)',Sigma(:,:,1:M,t-1));

        % Propagate the samples
        phi_xbar = zeros(size(phi_xhat));
        for ii = 1:N
            [phi_xbar(ii,:),~,~] =  dynfun(phi_xhat(ii,:));
        end

        % Fit GMM
        % note that M could be changed here.
        [w_new,mu_new,S_new,M] = sampleFitGMM(phi_xbar,M_max);
        w_bar(1:M,t) = w_new;
        x_bar(:,1:M,t) = mu_new';
        S(:,:,1:M,t) = S_new;
    end
    
    % 3. Meas Update -----------------------------------------------------
    w_t = zeros(1,M);
    for j = 1:M
        x_bar_jt = x_bar(:,j,t);
        S_jt = S(:,:,j,t);
        
        K = S_jt * C' / (C * S_jt * C' + K_V);
        Sigma(:,:,j,t) = (eye(n) - K*C) * S_jt;
        
        x_hat(:,j,t) = x_bar_jt + K * (y(:,t) - C*x_bar_jt);
        
        mu_post  = C * x_bar_jt;
        sig_post = C * S_jt * C' + K_V;
        
        w_t(j) = w_bar(j,t) * mvnpdf(y(:,t),mu_post,sig_post);
    end
    
    w_t = w_t/sum(w_t);
    w_hat(1:M,t) = w_t;
    
    GMMModel{t} = gmdistribution(x_hat(:,1:M,t)',Sigma(:,:,1:M,t),w_t);
    
    % state
    X_hat(:,t) = zeros(n,1);
    for j = 1:M
       X_hat(:,t) = X_hat(:,t) + w_t(j).*x_hat(:,j,t);
    end
    
    x_diff(:,t) = x_true(:,t) - X_hat(:,t);
    x_diff_norm(t) = norm(x_diff(:,t));
    
    M_size(t) = M;
    
    disp(['Time Step:',num2str(t-1),' |Xhat - X|:',num2str(x_diff_norm(t))]);
end


%% Postprocessing
isplot = 1;

% plot
if isplot
    close all;
    % W PDF
    figure(1)
    hold on;
    title('Noise Model');
    gmPDF = @(x,y) arrayfun(@(x0,y0) pdf(W_gm_true,[x0 y0]),x,y);
    fsurf(gmPDF,[-2 2])
    view([-134 18]);
    
    % PDF
    plot_ts = [1 5 10 15];
    figure(2)
    hold on;
    sgtitle('State Estimate Distribution');
    for ii = 1:length(plot_ts)
        subplot(2,2,ii)
        ti = plot_ts(ii);
        gmPDF = @(x,y) arrayfun(@(x0,y0) pdf(GMMModel{ti},[x0 y0]),x,y);
        fsurf(gmPDF,[-5 5])
        legend({['t: ',num2str(ti-1)]}, 'Location','northeast');
    end
    xlabel('X');
    ylabel('Y');
    zlabel('P(X,Y)');
    view([-134 18]);
    
    % Result
    figure(3);
    sgtitle('Results With FB Control')
    subplot(2,1,1)
    hold on;
    plot(x_true(1,:), x_true(2,:), 'ro-');
    plot(X_hat(1,:), X_hat(2,:), 'bo-');
    for ti = 1:T
        text(x_true(1,ti)+0.02, x_true(2,ti)+0.02, num2str(ti-1));
        text(X_hat(1,ti)+0.02, X_hat(2,ti)+0.02, num2str(ti-1));
    end
    xlabel('x');
    ylabel('y');
    legend({'True','Est'});
    
    subplot(2,1,2)
    plot(0:T-1, x_diff_norm, 'ko-');
    xlabel('TimeStep');
    ylabel('$$|\hat{x} - x|$$', 'Interpreter','latex');
end

%%
function [x_next, A, u] = dynfun1(x)
   % Duffling Oscillator
   a1 = 2.75;
   a2 = 0.2;
   
   u = zeros(2,1);
   x_next = [x(2); -a2*x(1) + a1*x(2) - x(2)^3] + u;  % next state (dyn + control)
   
   A      = [0 1; -a2, a1 - 3*x(2)^2];            % A - BU
end

function [x_next, A, u] = dynfun2(x)
   u      = [0; 0.3*x(1)*x(2)];
   x_next = [x(2); -1.1*x(1)*x(2)] + u; % next state (dyn + control)
   A      = [0 1; -0.8*x(2), -0.8*x(1)];   % A - BU
end


function phi = genSampleGMM(N,w,mu,Sigma)
    n = size(mu,2);
    M = length(w);
    phi = zeros(N,n); 
    
    w_cum = zeros(1,M);
    for j = 1:M
        w_cum(j) = sum(w(1:j));
    end

    for n = 1:N
        r = rand(1);
        sample_j = min(find(w_cum > r));
        phi(n,:) = mvnrnd(mu(sample_j,:),Sigma(:,:,sample_j),1);
    end
end

function [w,mu,Sigma,M] = sampleFitGMM(phi,M_max)

    % Run EM Algorithm for the realization -------------------------
    BIC = zeros(1,M_max);
    GMModel = cell(1,M_max);

    for j = 1:M_max
        is_err = 0;
        try
            GMModel{j} = fitgmdist(phi,j);  % fitting using EM Algorithm
        catch exception
            disp(['Fitting Error! M:',num2str(j)]);
            is_err = 1;
        end
        
        if ~is_err
            BIC(:,j) = GMModel{j}.BIC;
        else
            BIC(:,j) = Inf;
        end
    end
    [bestBIC,bestM] = min(BIC);

    w     = GMModel{bestM}.ComponentProportion;  % 1xM
    mu    = GMModel{bestM}.mu;                   % Mxn
    Sigma = GMModel{bestM}.Sigma;                % nxnxM

    M = bestM;  % chosse the one with best BIC
end