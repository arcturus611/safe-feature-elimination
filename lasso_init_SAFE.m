function [A, b, max_lambda, x0] = lasso_init_SAFE
%% Variables:
%% A: data matrix. each column is a feature, each row is a measurement. 
%% b: measurement vector
%% max_lambda: maximum l1-norm regularizer
%% lambda: l1-norm regularizer
%% x0: true vector  

load ee_578_s
rng(s);
m = 100; n = 300; k = 30;                  % No. of rows (m), columns (n), and nonzeros (k) 
A = randn(m, n);
A = A/10;                                   % ... A is m-by-n
p  = randperm(n); p = p(1:k);              % Location of k nonzeros in x
x0 = zeros(n,1); x0(p) = randn(k,1);       % The k-sparse solution
n = .02*randn(m, 1);
b  = A*x0 + n;               % add random noise   
max_lambda = norm( A'*b, 'inf' );
end