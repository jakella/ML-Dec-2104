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
a_1 = [ones(m, 1), X];
size(X)
Theta1_size = size(Theta1)
Theta2_size = size(Theta2)
y_t = zeros(m, num_labels);
       
% You need to return the following variables correctly 
J = 0; regular1 = 0; regular2 =0;
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
%
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
%
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

a2 =  sigmoid(a_1*Theta1');
a_2 = [ones(m, 1), a2];
a3 = sigmoid(a_2*Theta2');
Theta1_r = Theta1(:, [2:end]);
Theta2_r = Theta2(:, [2:end]);


disp("Starting the for loop over samples");

for i=1:m

  y_t(i,y(i,1)) = 1;

  %delta_3 = a3(i,:) - y_t(i,:)
  %delta3_size = size(delta_3)
  
  %theta1_size = size(Theta1)
  
  %delta_2 =  (delta_3*(Theta2_r)).*sigmoidGradient(a_1(i,:)  *Theta1');
  %delta_2 = delta_2(2:end);

  %size_delta_2 = size(delta_2)
  %size_a_1 = size(a_1(i,:))
  %size_delta_3 = size(delta_3)
  %size_a_2 = size(a_2(i,:))

  %size_Theta2_grad  = size(Theta2_grad)
  %size_Theta1_grad  = size(Theta1_grad)


  %Theta2_grad = Theta2_grad + delta_3'*a_2(i,:);
  %Theta1_grad = Theta1_grad + delta_2'*a_1(i,:);


  J = J + -(1/m)*( y_t(i,:)*log(a3(i,:))'+
           (1-y_t(i,:))*log(1-(a3(i,:)))');

endfor

for j=1:hidden_layer_size
  for k=1:input_layer_size 
   
   regular1 = regular1 + Theta1_r(j,k)^2;
endfor
endfor

 
for p=1:num_labels
  for q=1:hidden_layer_size 
   
   regular2 = regular2 + Theta2_r(p,q)^2;
endfor
endfor

regular1
regular2
m
lambda

regular = (lambda/(2*m))*(regular1+regular2)

J = J + regular


delta_3 = a3 - y_t;
delta_2 =  delta_3*Theta2_r.*sigmoidGradient(a_1*Theta1');


 Theta2_grad = (1/m)*(Theta2_grad + delta_3'*a_2);
 Theta1_grad = (1/m)*(Theta1_grad + delta_2'*a_1);

regular_theta2 = (lambda/m)*Theta2;
regular_theta2 = regular_theta2(:,2:end);
regular_theta2= [zeros(size(regular_theta2,1), 1), regular_theta2];

Theta2_grad = Theta2_grad  + regular_theta2

regular_theta1 = (lambda/m)*Theta1;
regular_theta1 = regular_theta1(:,2:end);
regular_theta1 = [zeros(size(regular_theta1,1), 1), regular_theta1];

Theta1_grad = Theta1_grad  + regular_theta1



% -------------------------------------------------------------
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)]


end
