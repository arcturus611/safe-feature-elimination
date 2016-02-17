%% Feature Elimination problem for LASSO
clear all; close all; clc;
digits(10);

%% Set up problem
[A, b, max_lambda, x0] = lasso_init_SAFE;
At = A';
num_features = size(A, 2);

num_lambda_experiments = 20;
lambda_vec = max_lambda*linspace(.001, 1, num_lambda_experiments);

num_gamma_experiments = 10;

SAFE_output_struct = struct('solution', zeros(num_features, 1), 'error_l2', [], 'solution_l1', [], ...
    'elim_features', zeros(num_features, num_gamma_experiments), 'num_elim_features', zeros(1, num_gamma_experiments), ...
    'missed_elim_features', zeros(num_features, num_gamma_experiments), 'num_missed_elim_features', zeros(1, num_gamma_experiments), ...
    'incorrect_elim_features', zeros(num_features, num_gamma_experiments), 'num_incorrect_elim_features', zeros(1, num_gamma_experiments));
SAFE_output = repmat(SAFE_output_struct, num_lambda_experiments, 1);
epsilon = 2*10^(-4); % used for polishing CVX solution

%% Solve problem using CVX and eliminate features using SAFE rule
for ii = 1 :num_lambda_experiments
    ii
    lambda = lambda_vec(ii);
    dual_feasible_point = -lambda*b/norm( A'*b ,'inf');
    lower_bound_max = -.5*dual_feasible_point'*dual_feasible_point - b'*dual_feasible_point;
    gamma_vec = lower_bound_max*linspace(0.001, 1, num_gamma_experiments);
    
    %% Solve problem by CVX
    cvx_begin quiet
    cvx_precision low
    variable xCVX(num_features)
    minimize 0.5*sum_square(A*xCVX - b) + lambda*norm(xCVX, 1)
    cvx_end
    
    %% Polish CVX solution
    xCVX(abs(xCVX)<epsilon) = 0;
    
    SAFE_output(ii).solution = xCVX;
    SAFE_output(ii).error_l2 = norm(A*xCVX - b, 2)^2;
    SAFE_output(ii).solution_l1 = norm(xCVX, 1);
    
    for jj = 1:num_gamma_experiments
        gamma = gamma_vec(jj);
        
        %% SAFE_output rule (equation 10/11 of SAFE_output paper)
        alg_const = sqrt(b'*b - 2*gamma); % used in equation 10
        for i = 1:num_features
            xk = A(:, i);
            if( lambda>( abs(xk'*b) + alg_const*norm(xk, 2) ) ) %equation 10 SAFE_output
                SAFE_output(ii).elim_features(i, jj) = 1;
            end
        end
        SAFE_output(ii).num_elim_features(jj) = sum(SAFE_output(ii).elim_features(:, jj));
        
        %% Check how well SAFE_output test performed.
        SAFE_output(ii).missed_elim_features(:, jj) = ( (~xCVX) & (~SAFE_output(ii).elim_features(:, jj)) );
        SAFE_output(ii).incorrect_elim_features(:, jj) = ( (xCVX)  &  (SAFE_output(ii).elim_features(:, jj))  );
        
        SAFE_output(ii).num_missed_elim_features(jj) = sum(SAFE_output(ii).missed_elim_features(:, jj));
        SAFE_output(ii).num_incorrect_elim_features(jj) = sum(SAFE_output(ii).incorrect_elim_features(:, jj));
        
    end
end

%% Draw the Pareto optimal curve
error_l2 = [SAFE_output.error_l2];
solution_l1 = [SAFE_output.solution_l1];
figure, plot(error_l2, solution_l1), title('pareto optimal curve');
ylabel('l1-norm of solution');
xlabel('l2-norm of error');
%% Plot number of features eliminated for each lower bound
num_elim_features_v = [SAFE_output.num_elim_features];
num_elim_features_m = reshape(num_elim_features_v, [num_gamma_experiments, num_lambda_experiments]);
plot_colors = jet(num_lambda_experiments);
figure;
hold on; grid on;
h = zeros(1, num_lambda_experiments);
for kk = 1:num_lambda_experiments
    h(kk) = plot(gamma_vec, num_elim_features_m(:, kk), 'Color', plot_colors(kk, :), 'DisplayName', sprintf('lambda= %.2f', lambda_vec(kk)));
end
ax = gca;
set(ax, 'XTick', gamma_vec);
legend(h);
title('Effect of l1-norm regularizer and primal lower bound on feature elimination');
xlabel('\gamma'); ylabel('number of features eliminated');

%% Number of incorrect features eliminated
fprintf('The code incorrectly eliminated features %d times\n', sum([SAFE_output(:).num_incorrect_elim_features]));
inc_elim_fea_v = [SAFE_output(:).num_incorrect_elim_features];
inc_elim_fea_m = reshape(inc_elim_fea_v, [num_gamma_experiments, num_lambda_experiments]);

%% Plot solution, eliminated features, missed features and incorrectly eliminated features
feature_vec = 1:num_features;
figure;
hold on;
plot_lambda = 17;
plot_gamma = num_gamma_experiments-1;
elim = SAFE_output(plot_lambda).elim_features(:, plot_gamma);
num_elim = SAFE_output(plot_lambda).num_elim_features(plot_gamma);
missed = SAFE_output(plot_lambda).missed_elim_features(:, plot_gamma);
num_missed = SAFE_output(plot_lambda).num_missed_elim_features(plot_gamma);
inc = SAFE_output(plot_lambda).incorrect_elim_features(:, plot_gamma);
num_inc = SAFE_output(plot_lambda).num_incorrect_elim_features(:, plot_gamma);
soln = SAFE_output(plot_lambda).solution;
h2 = [];
h3 = [];
h4 = [];
h1 = plot(feature_vec, soln, 'b', 'DisplayName', 'Solution');
if(num_elim)
    h2 = plot(feature_vec(elim>0), ~elim(elim>0), 'ko', 'MarkerSize', 2, 'MarkerFaceColor', 'k','DisplayName', 'Eliminated Features');
end
if (num_missed)
   h3 = plot(feature_vec(missed>0), ~missed(missed>0), ...
     'ro', 'MarkerSize', 2, 'MarkerFaceColor', 'r', 'DisplayName', 'Missed Features');
end
if (num_inc)
   h4 = plot(feature_vec(inc>0), ~inc(inc>0), 'g*', 'DisplayName', 'Incorrectly Eliminated Features');
end

legend([h1 h2 h3 h4]);
title(['CVX and feature elimination output outputs with \lambda = ' num2str(lambda_vec(plot_lambda)) ' and \gamma = ' num2str(gamma_vec(plot_gamma))]);
