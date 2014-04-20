function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%this part is so similar that I think one might be able to use the
%code in costFunction.m directly; copy/paste for now
%%%%SUBMITTED CODE
%for i=1:m
    %here is the standard part
%    h_theta_i = sigmoid(theta' * X(i,:)' );
%    J = J - y(i) * log( h_theta_i ) - (1 - y(i))*log(1 - h_theta_i);
%    for jj=1:length(grad)
%	grad(jj) = grad(jj) + (h_theta_i - y(i))*X(i,jj);
%    end
%end

%%%%NEW CODE
h = sigmoid(X * theta);
J = sum(-y .* log( h ) - (1-y) .* log(1- h));

%%%NEW CODE FOR GRADIENT
#grad = sum( repmat(h - y,1,size(X)(2)) .* X , 1);
#grad = grad';

%%%EVEN NEWER CODE FOR GRADIENT
%based on the problem set notes
% deriv = 1/m X' * (h-y)
grad = X' * (h-y);

%%%%SUBMITTED CODE -- no need to change it
%now add cost function penalty terms, excluding theta_1
theta_reduced = theta(2:length(theta));
J = J + 0.5 * lambda * sum( theta_reduced .^ 2) ;
J = J / m;

%now add penalty part to grad
theta(1) = 0; %no penalty for first term
grad = grad + lambda * theta;

grad = grad / m;

% =============================================================

end
