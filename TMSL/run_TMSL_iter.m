%% Load Setting
clear;
addpath([pwd, '/funs']);
addpath([pwd, '/Nuclear_norm_l21_Algorithm']);
addpath([pwd, '/ClusteringMeasure']);
addpath(genpath('proximal_operator'));
addpath(genpath('tSVD'));
addpath(genpath('utils'));
addpath(genpath('Ncut_9'));

dataname='MSRC_v1';  %%ORL MSRC_v1 EYaleB10 COIL20 BBCSport Caltech101_20 scene15 UCI 3sources
load(strcat('../data/',dataname,'.mat'));
lambda1=1e-2;  
lambda2=1e-2;
lambda3=1e-4;
DEBUG = 1;
max_iter = 30;
epson = 1e-7; max_mu = 10e10; pho_mu = 2;
fprintf('start: %s, lambda1=%.4f,lambda2=%.4f,lambda3=%.4f:\n',dataname,lambda1,lambda2,lambda3);

%% preparation
cls_num = length(unique(gt));
T = length(X); N = size(X{1},2); %sample number
sX = [N, N, T];


%% Optimizataion
tic;
mu = 10e-5; 
iter = 0;

%% Initialize and Settings
for t=1:T
    Z{t} = zeros(N,N);

    E1{t} = zeros(size(X{t},1),N);
    Y1{t} = zeros(size(X{t},1),N);

    PX{t} = inv(X{t}' * X{t} + 2*eye(N)); %for accelerate 
end

S = zeros(N,N,T);
J=S;
L=S;
K=S;
Y2=S;
Y3=S;
Y4=S;
Y5=S;
E2=S;
It = S; %identity tensor
It(:,:,1) = eye(N);

while iter < max_iter
    for t=1:T
        %% Update Z^t
        Z{t} = PX{t}*(X{t}'*(X{t}-E1{t}+Y1{t}/mu) + J(:,:,t)-Y3(:,:,t)/mu + L(:,:,t)-Y4(:,:,t)/mu);

        %% Update E1^t
        F = [];
        for k1=1:T
            F = [F;X{k1}-X{k1}*Z{k1}+Y1{k1}/mu];
        end
        [Econcat] = solve_l1l2(F,lambda1/mu);
        e_start = 1; e_end = 0;
        for k1 = 1:T
            e_end = e_end + size(X{k1},1);
            E1{k1} = Econcat(e_start:e_end, :);
            e_start = e_start + size(X{k1},1);
        end

       %% Update Y1^t
        Y1{t} = Y1{t} + mu*(X{t}-X{t}*Z{t}-E1{t});
    end

    Zt = cat(3, Z{:,:});
    %% Update S
    S1 = tprod(tran(L),L-E2+Y2/mu);
    S = tprod(t_inverse(L),S1+K-Y5/mu);

    %% update E2
    E2 = prox_l1(L-tprod(L,S)+Y2/mu, lambda3/mu );

    %% update J
    z = Zt(:);
    y3 = Y3(:);
    [j, ~] = wshrinkObj(z + 1/mu*y3,1/mu,sX,0,3);
    J = reshape(j, sX);
%     [J,~] = prox_tnn(Zt+Y3/mu,1/mu);

    %% update K
    s = S(:);
    y5 = Y5(:);
    [k, ~] = wshrinkObj(s + 1/mu*y5,lambda2/mu,sX,0,3); 
    K = reshape(k, sX);

%     T1 = shiftdim(S+Y5/mu, 1);
%     [TK,~] = prox_tnn(T1,1/mu);
%     K = shiftdim(TK, 2);

    %% update L
    M = It-S;
    L1 = tprod(E2-Y2/mu,tran(M));
    L = tprod(L1+Zt+Y4/mu,t_inverse2(M));

    %% check convergence
    for t=1:T
        e1(t) = norm(X{t}-X{t}*Z{t}-E1{t},inf);
    end
    em1 = max(e1);

    leq2 = L-tprod(L,S)-E2;
    leqm2 = max(abs(leq2(:)));

    leq3 = Zt-J;
    leq4 = Zt-L;
    leq5 = S-K;

    leqm3 = max(abs(leq3(:)));
    leqm4 = max(abs(leq4(:)));
    leqm5 = max(abs(leq5(:)));

    err = max([em1,leqm2,leqm3,leqm4,leqm5]);
%     if DEBUG && (iter==1 || mod(iter,2)==0)
%         fprintf('iter = %d, err = %.8f \n',iter,err)
    if DEBUG
        fprintf('iter = %2d, %4.8f %4.8f %4.8f %4.8f %4.8f\n',iter,em1,leqm2,leqm3,leqm4,leqm5);;
    end
    if err < epson
        break;
    end

   %% update Lagrange multiplier and  penalty parameter beta
    Y2 = Y2 + mu*leq2;
    Y3 = Y3 + mu*leq3;
    Y4 = Y3 + mu*leq4;   
    Y5 = Y3 + mu*leq5;

    mu = min(mu*pho_mu, max_mu);
    iter = iter + 1;
end

Aff = 0;
for t=1:T
    Aff = Aff + abs(K(:,:,t))+abs(K(:,:,t)');
end
if(strcmp(dataname, 'COIL20'))
    Aff = double(Aff);
end
for index = 1: 10
    clu = SpectralClustering(Aff./T, cls_num);
    
    [A nmi(index) avgent] = compute_nmi(gt,clu);
    ACC(index) = Accuracy(clu,double(gt));
    [f(index),p(index),r(index)] = compute_f(gt,clu);
    [AR(index),RI,MI,HI]=RandIndex(gt,clu);
    fprintf('%d: %.4f %.4f %.4f %.4f %.4f %.4f\n',index,ACC(index),nmi(index),AR(index),f(index),p(index),r(index));
    toc;
end
fprintf('upa-RTMVC,%.3f¡À%.3f,%.3f¡À%.3f,%.3f¡À%.3f,%.3f¡À%.3f,%.3f¡À%.3f,%.3f¡À%.3f \n',...
    mean(ACC),std(ACC),mean(nmi),std(nmi),mean(AR),std(AR),mean(f),std(f),mean(p),std(p),mean(r),std(r));