function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

%model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));

Cv = [0.01 0.03 0.1 0.3 1 3 10 30];
sv = [0.01 0.03 0.1 0.3 1 3 10 30];

min_error = 1000000;
besti = 0;
bestj = 0;

for i=1:length(Cv) %loop over C
  for j=1:length(sv) %loop over sigma
      thisC = Cv(i);
      thisS = sv(j);
      %train on training data
      %%%what about these x1 and x2 parameters?
      model= svmTrain(X, y, thisC, @(x1, x2) gaussianKernel(x1, x2, thisS));
      %test on Validation data
      pred = svmPredict(model,Xval);
      thisError = mean(double(pred ~= yval));
      %evaluate performance
      printf("C = %f, sigma = %f ; error rate = %f\n",thisC,thisS,thisError);
      if ( thisError < min_error )
         min_error = thisError;
	 besti = i;
	 bestj = j;
      endif
  endfor
endfor

C = Cv(besti);
sigma = sv(bestj);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


% =========================================================================

end
