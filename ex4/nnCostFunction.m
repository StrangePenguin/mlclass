function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

%need to append bias to X
a1 = [ones(m,1)  X];
%we really want a column vector for each example
a1 = a1';

%%%%%vectorized version of J calculation
z2_v = Theta1 * a1;
a2_v = sigmoid(z2_v);
%add bias row
a2_vb = [ones(1,m) ; a2_v];
a3_v = sigmoid(Theta2 * a2_vb);
%not sure how to get rid of the for loop in defining the y matrix
y_v = zeros(num_labels,m);
for i=1:m
    y_v( y(i), i) = 1;
end
J_v = -y_v .* log( a3_v ) - (1-y_v) .* log(1-a3_v);
J_v_sum = sum( J_v(:));
% Works!
%%% vectorized version of backprop calculation
delta3_v = a3_v - y_v;
delta2_v = Theta2' * delta3_v;
%drop bias term
delta2_vnb = delta2_v(2:end,:);
delta2_z2 = delta2_vnb .* sigmoidGradient( z2_v );
Theta2_grad = delta3_v * a2_vb';
Theta1_grad = delta2_z2 * a1';
%%%% end

%for i=1:m
%  %extract this training example
%  a1i = a1(:,i); %column vector
%  z2 = Theta1 * a1i;
%  a2 = sigmoid( z2 );
%  a2 = [1 ; a2]; %add bias
%  a3 = sigmoid( Theta2 * a2);
%%now y
%  thisy = zeros(num_labels,1);
%  thisy( y(i) ) = 1;
%%and now the sum over K
%  %J = J + sum(-thisy .* log( a3 ) - (1-thisy) .* log(1 - a3));
%%%%%%%now the backprop part
%  delta3 = a3 - thisy;
%  delta2 = (Theta2' * delta3);
%  %drop term associated with bias
%  delta2 = delta2(2:end);
%  delta2 = delta2 .* sigmoidGradient( z2);
%  Theta2_grad = Theta2_grad + delta3 * a2';
%  Theta1_grad = Theta1_grad + delta2 * a1i';
%end


%printf('Before regularization, J values are %f %f\n',J/m,J_v_sum/m);
%looks good!

%now add regularization
%%%important -- drop the first column of each matrix
Theta1r = Theta1(:,2:input_layer_size+1);
Theta2r = Theta2(:,2:size(Theta2)(2));
nn_params_r = [Theta1r(:) ; Theta2r(:)];
reg_term =0.5*lambda*sum( nn_params_r .^ 2); 
%J = J + reg_term;
J_v_sum = J_v_sum + reg_term;

%%gradient regularization
%%%put back the first column but as zeroes
Theta1rg = [zeros(size(Theta1)(1),1)  Theta1r];
Theta2rg = [zeros(size(Theta2)(1),1)  Theta2r];
Theta1_grad = Theta1_grad + lambda * Theta1rg;
Theta2_grad = Theta2_grad + lambda * Theta2rg;

%J = J/m;
J = J_v_sum/m;

%printf('J values are %f %f\n',J,J_v_sum);

Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.


%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
