function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%X has ones already added; needs to be transposed
XT = X';
h= theta' * XT;

diff = h'-y;

J = sum( diff.^2 );

%now the gradient
%%this is non-obvious, i would say
grad = (diff' * XT')';

%now add regularization

theta(1)=0;
J = J + lambda*sum(theta.^2);
grad = grad + lambda*theta;

%divide by m
J=0.5*J/m;
grad = grad /m;

% =========================================================================

grad = grad(:);

end
