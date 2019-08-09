function [Y,obj,Q]=SPSL(varargin)
    %%=============================================
    %INPUT:
    % X: the input data DxN
    % params£ºparameter structure
    % label or not ? 
    %params example
    % params = struct('maxiter', 50, ...
    %                 'no_dim',3,... dimensionality
    %                 'eps', 1e-5,  For ADMM
    %                 'gstruct', 'l1' or 'l2'
    %                 'lambda', 1.0, ... the bound of a_ij
    %                 'sd_rate', 0.5, ...
    %                 'lambda_2', 0.01, ...
    %                 'gamma', Fixed, ...
    %                 'verbose',true,... For ADMM
    %                 'drawnow' true   plot the result in each step
    %                  'knn',    10,...)
    %OUTPUT:
    % Z: the data in low-dimensionality
    % Q: kernel matrix  learned  from our method
    %create by HJ W 2019.8.8
    %Data Visualization via Expectation Of Distance Preserved
    %%=============================================
    %%
    labels=[];
    
    if length(varargin)<2
        error('input isn''t enough');
    end
    
    if length(varargin)==3 
        labels=varargin{3};
        knn=-1;
        cls = unique(label);
        for i=1:length(cls)
            idx = find(label==cls(i));
            S(idx, idx) = 1;
        end
    end
    if length(varargin)>3 
        error('too much input');
    end
    if length(varargin)==2 
        knn=varargin{2}.knn;
    end

    X = varargin{1};
    [n,d]=size(X)
    S = zeros(n, n);
    params=varargin{2};
    global verbose  sqdist idx_map;
    maxiter	=params.maxiter ;
    no_dims	=params.no_dims ;
    eps		=params.eps     ;
    gstruct	=params.gstruct ; %'l1' or 'l2'
    lambda_1=params.lambda_1; %the bound of a_ij
    store   =params.store   ; %how much steps shoud ADMM stroe; 
    sd_rate	=params.sd_rate ; % the weight between S and D
    lambda_2=params.lambda_2; 
    gamma	=params.gamma   ; %fixed with 1
    verbose	=params.verbose ;
    drawnow	=params.drawnow ; %plot the result in each step
    knn		=params.knn     ; %use labels or not
    rho     =params.rho     ;
    if verbose
        ADMM_print=10;
    else
        ADMM_print=inf;
    end
    % initialization
    normsq = repmat(sum(X.^2,2), 1, n);
    sqdist = (normsq + normsq' - 2 .* X * X')./d; % normalized by dim usually good
    if knn <= 0 
    % fully connected graph
    graph = sparse(tril(ones(n,n) - eye(n,n)));
    else
    % k-nearest neighbor graph
    [~, sort_idx]=sort(sqdist, 2, 'ascend');
    knn_idx = sort_idx(:,1:(knn+1));
    for i=1:n
        S(i,knn_idx(i,:))=1;
    end
    knnG=sparse(repmat(knn_idx(:,1),knn,1), reshape(knn_idx(:,2:(knn+1)),...
        n*knn, 1),ones(n*knn,1), n, n);
    graph = max(knnG, knnG');
    end

    D = ones(n, n) - S;
    LS = diag(sum(S)) - S;
    LD = diag(sum(D)) - D;
    len_S = sum(sum(S));
    len_D = sum(sum(D));
    LSD = (len_S+len_D) .* ( sd_rate .* LS ./ len_S + (1-sd_rate) .* LD ./len_D);

    
    [rows, cols] = find(graph);
    idx_map = [rows, cols];

    m = size(idx_map,1);
    l = zeros(m,1);
    u = lambda_1 .* ones(m, 1);

    A = sparse(n, n);
    R = eye(n, n);

    % ADMM iterations
    LA = diag(sum(A)) - A;
    C = eye(n, n) + 4 * gamma .* (lambda_2 .* LSD + LA) + R ./ rho;
    %%
    for iter=1:maxiter
        % update Q
        old_C = C;
        [eig_vec, eig_val] = mexeig(C);
        eig_val = diag(eig_val ./ 2);
        new_eig_val = diag(eig_val + sqrt(eig_val.^2 + 0.5 * no_dims * ones(n, 1)./rho));
        Q = eig_vec * new_eig_val * eig_vec';
        
        % uodate A
        P = Q - eye(n,n) - 4 * lambda_2 * gamma .* LSD - R ./ rho;
        P = P ./ (4 * gamma);
        
        fobj = @(a)compute_objective(a, P, rho, gamma);
        fgrad = @(a)compute_gradient(a, P, rho, gamma);
        fun = @(x)fminunc_wrapper(x, fobj, fgrad);
        opts = struct( 'factr', 1e4, 'pgtol', eps, 'm', store, 'printEvery', ADMM_print); 
        if m > 10000; opts.m = 50; end
        [a, ~, ~] = lbfgsb(fun, l, u, opts);
        A = sparse(idx_map(:,1), idx_map(:,2), a, n, n);
        A = full(A + A');
        
        % update multiplier
        LA = diag(sum(A)) - A;
        delta_R = Q - eye(n, n) - 4 * gamma .* (lambda_2 .* LSD + LA);
        R = R - rho .* delta_R;
        
        % update rho
        C = eye(n, n) + 4 * gamma .* (lambda_2 .* LSD + LA) + R ./ rho;
        norm_r = norm(delta_R, 'fro');
        norm_s = norm(rho .*(C - old_C), 'fro');
        if norm_r > 10 * norm_s
            rho = 2 * rho;
        elseif norm_s > 10 * norm_r
            rho = rho / 2;
        end
        
        % compute objective value
        [B, p] = chol(Q);
        obj(iter) = -no_dims*sum(log(diag(B))) + trace(A' * sqdist);
        fprintf('iter=%d, obj=%f\n', iter, obj(iter));
        if drawnow
            S = inv(Q);
            column_sums = sum(S) / n;
            J = ones(n, 1) * column_sums;
            K = S - J - J' + sum(column_sums)/n;
    
            % eigendecomposition
            [U, V] = eig(K);
            [v, v_ind] = sort(diag(V), 'descend');
            v = v(1:no_dims);
            U = U(:, v_ind(1:no_dims));
            Y = U * diag(sqrt(v));

            if ~isempty(labels)
                if no_dims == 1
                    scatter(Y(:,1), 9, labels, 'filled');
                elseif no_dims == 2
                    scatter(Y(:,1), Y(:,2), 9, labels, 'filled');
                else
                    scatter3(Y(:,1), Y(:,2), Y(:,3), 40, labels, 'filled');
                end
                axis equal tight
    %             axis off
                drawnow;
            else
                if no_dims == 1
                    scatter(Y(:,1), 9, 'filled');
                elseif no_dims == 2
                    scatter(Y(:,1), Y(:,2), 9, 'filled');
                else
                    scatter3(Y(:,1), Y(:,2), Y(:,3), 40, 'filled');
                end
                axis equal tight
    %             axis off
                drawnow;

            end


        end
    end

       % centeralize kernel
        S = inv(Q);
        column_sums = sum(S) / n;
        J = ones(n, 1) * column_sums;
        K = S - J - J' + sum(column_sums)/n;

        % eigendecomposition
        [U, V] = eig(K);
        [v, v_ind] = sort(diag(V), 'descend');
        v = v(1:no_dims);
        U = U(:, v_ind(1:no_dims));
        Y = U * diag(sqrt(v));

end

function obj = compute_objective(a, P, rho, gamma)
global sqdist idx_map
n = size(P, 1);
A = sparse(idx_map(:,1), idx_map(:,2), a, n, n);
A = full(A + A');
LA = diag(sum(A)) - A;
obj = sum(sum(A .* sqdist)) + 8 * rho * gamma^2 * sum(sum((LA - P).^2));
% fprintf('obj=%f\n',obj);    

end

function grad = compute_gradient(a, P, rho, gamma)
global sqdist idx_map verbose
n = size(P, 1);
A = sparse(idx_map(:,1), idx_map(:,2), a, n, n);
A = full(A + A');
LA = diag(sum(A)) - A;
U = LA - P;
M = U + U' - diag(diag(U));

m = length(a);
grad = zeros(m, 1);
for k=1:m
    i = idx_map(k,1);
    j = idx_map(k,2);
    tmp = M(i,i) + M(j,j) - M(i,j) - M(j,i);
    grad(k) = 16 * rho * gamma^2 * tmp + 2 * sqdist(i,j);
end
if verbose
fprintf('.')
end
end