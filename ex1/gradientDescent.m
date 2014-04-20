function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %


    %thetanew_j = theta_j - alpha*(1/m) *sum_1..m ( htheta(xi)-yi)*xi_j
    d = zeros(length(theta),1);
    for j=1:length(theta)
      for i=1:m
        thisSet = X(i,:); %no need to append 1; done already
        h = theta' * thisSet'; 
        d(j) = d(j) + ( h - y(i))*thisSet(j);
      end
    end
    theta = theta - (1/m)*alpha*d;


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
