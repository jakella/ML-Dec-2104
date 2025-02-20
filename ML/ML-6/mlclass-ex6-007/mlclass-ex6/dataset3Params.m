function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

C_vec = [0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';
sigma_vec = [0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

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

for i = 1:length(C_vec)
  for j = 1:length(sigma_vec)
     C_test = C_vec(i);
     sigma_test = sigma_vec(j);
	model= svmTrain(X, y, C_test, @(x1, x2) gaussianKernel(x1,  
             x2,sigma_test)); 
     predictions = svmPredict(model, Xval);
     svm_error(i,j) = mean(double(predictions ~= yval));

end
end


mm = min(min(svm_error));
[k, l] = find(svm_error==mm);
 C = C_vec(k)
 sigma  = sigma_vec(l)



% =========================================================================

end
