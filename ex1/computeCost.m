function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

  % J (theta) = 1/2m *
  % sum_i=1..m (htheta(xi) - yi )^2
% htheta = theta0*1 + theta1*x1 +...

%loop over rows from 1 to m
for i = 1:m
  thisSet = X(i,:); %no need to append 1; done already
  h = theta' * thisSet'; 
  J = J + ( h - y(i))^2;
end

J = J/(2*m);

% =========================================================================

end
