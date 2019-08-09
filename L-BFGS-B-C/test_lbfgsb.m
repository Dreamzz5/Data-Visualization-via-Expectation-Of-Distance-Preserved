function test_lbfgsb
addpath('Matlab')
global A b

N   = 1000; 
M = 1500;     
A   = randn(M,N);
b   = randn(M,1);

x = [];
time = [];

%% Solve NNLS with L-BFGS-B

l  = zeros(N,1);    % lower bound
u  = inf(N,1);      % there is no upper bound
tstart=tic;
fun     = @(x)fminunc_wrapper( x, @fcn, @gradient); 
% Request very high accuracy for this test:
opts    = struct( 'factr', 1e4, 'pgtol', 1e-8, 'm', 10);
opts.printEvery     = 5;
if N > 10000
    opts.m  = 50;
end
% Run the algorithm:
[xk, ~, info] = lbfgsb(fun, l, u, opts );
t=toc(tstart)
% Record results
x.lbfgsb    = xk;
time.lbfgsb = t;


function grad = gradient( x )
global A b

AtA     = A'*A; Ab = A'*b;
grad    = 2*( AtA* x - Ab );

function obj = fcn(x)

global A b
obj = norm( A*x - b)^2;