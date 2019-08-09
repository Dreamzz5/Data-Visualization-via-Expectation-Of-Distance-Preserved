function experiment_toy
addpath(genpath('./'))
filenames = {'Circle','two_moon','tree_300','Spiral','three_clusters','DistortedSShape'};
%for i=1:length(filenames)
for i=1:1
    filename = filenames{i};

    % l1-graph
    eps_filename = sprintf('results/l1_%s.eps',filename);
    params =struct( 'maxiter' ,20   ,...
                       'no_dims'  ,2    ,...
                       'eps'     ,1e-4 ,...
                       'gstruct' ,'l2' ,...
                       'lambda_1',1    ,...
                       'store'   ,15   ,...
                       'sd_rate' ,0.8  ,...
                       'lambda_2',1e-4 ,...
                       'gamma'   ,1    ,...
                       'verbose' ,false,...
                       'drawnow' ,false ,...
                       'rho'     ,1    ,...
                       'knn'     ,10 );
                       
    
    run_our_method(filename, eps_filename, params);
    
    
end

function run_our_method(filename, eps_filename, params)

switch (filename)
    case 'Circle'
        % circle
        load('Circle'); 
        
    case 'two_moon'
        % two_moon
        load('two_moon'); 
        
    case 'tree_300'
        % tree
        load('tree_300'); 
            
    case 'Spiral'
        % spiral
        load('Spiral_SUN'); 
        
    case 'three_clusters'
        % three_clusters
        load('three_clusters'); 
        
    case 'DistortedSShape'
        % distrotedSShape
        load('DistortedSShape'); 
        
    otherwise
        warning('unexpected data settings for %s', name);
        return
end

% run the proposed method
start_time = cputime;
[Y,objs,Q]=SPSL(X,params);
elapse_time = cputime - start_time;
fprintf('time cost is: %f sec\n',elapse_time);
figure;
plot(Y(:,1), Y(:,2), 'bo','MarkerSize',8);
title(filename)
figure;
plot(X(:,1), X(:,2), 'bo','MarkerSize',8);
title(strcat('Orignal_',filename));
% plot results
%plot_graph_toy(X, C, W, eps_filename)

%% print convergence
% h = figure; 
% plot(objs,'LineWidth',2);
% set(gca, 'FontSize',16);
% xlabel('iteration')
% ylabel('objective value');

%converge_filename = sprintf('results/converge/converge_%s.eps',filename);
%print(h, '-depsc',  converge_filename);
%close(h);

function plot_graph_toy(X, C, W, eps_filename)

W(W <1e-5) = 0;

[iidx, jidx, val] = find(sparse(W));
% fprintf ('%d, %d, %f\n', [iidx, jidx, val]');
h=figure;
box on;
hold on;
for i=1:length(iidx)
    if ~isnan(val(i))
        plot( [C(1, iidx(i)), C(1, jidx(i))], [C(2, iidx(i)), C(2, jidx(i))],...
            'ko-','LineWidth',2,'MarkerSize',10,'MarkerEdgeColor','k' );
    end
end
plot(C(1,:), C(2,:),'ko');
plot(X(1,:), X(2,:), '.b','MarkerSize',8);
xlim([ min(X(1,:)), max(X(1,:)) ]);
ylim([ min(X(2,:)), max(X(2,:)) ]);
set(gca, 'FontSize',16);

print(h, '-depsc',  eps_filename);

