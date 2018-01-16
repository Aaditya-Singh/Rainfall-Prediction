Training_Input = zeros(120,5819);
for i =1:5819
    mu = mean(training_set_input_5(:,i));
    sigma = std(training_set_input_5(:,i));
    if sigma == 0
        Training_Input(:,i) = zeros(120,1);
        continue
    end
    for j = 1:120
    Training_Input(j,i) = (training_set_input_5(j,i)-mu)/sigma;
    end
end
Quad_Training_Input = ones(120,1);
for i = 1:5819
    Quad_Training_Input = [Quad_Training_Input,Training_Input(:,i),Training_Input(:,i).^2];
end
Test_Input = zeros(26,5819);
for i =1:5819
    mu = mean(test_set_input_5(:,i));
    sigma = std(test_set_input_5(:,i));
    if sigma == 0
        Test_Input(:,i) = zeros(26,1);
        continue
    end
    for j = 1:26
    Test_Input(j,i) = (test_set_input_5(j,i)-mu)/sigma;
    end
end
Quad_Test_Input = ones(26,1);
for i = 1:5819
    Quad_Test_Input = [Quad_Test_Input,Test_Input(:,i),Test_Input(:,i).^2];
end
% Training %
m = 11639;                                          %% No. of parameters
X = Quad_Training_Input;                            %% size(X) = 120*11639
Y = training_output;                                %% size(Y) = 120*1
theta = zeros(m,1);                                 %% size(theta) = 11639*1
% For regularization
lambda = 1000000;
J_prev = (X*theta-Y)'*(X*theta-Y)/(2*m) + (lambda/(2*m))*theta(2:m)'*theta(2:m); 

% choose initial epsilon
epsilon = 1;                                        

% choose epsilon_min
epsilon_min = 0.001;

% choose learning rate
alpha = 0.001;

while epsilon>=epsilon_min
    % Gradient Descent
    theta = theta - (X'*(X*theta-Y) + lambda*[0;theta(2:m)])*(alpha/m) ;
    J_curr = (X*theta-Y)'*(X*theta-Y)/(2*m) + (lambda/(2*m))*theta(2:m)'*theta(2:m);
    epsilon = 100*abs(J_curr-J_prev)/J_curr;
    J_prev = J_curr;
end

% Testing %
Predicted_test_output = Quad_Test_Input*theta ;
test_error = 100*abs(test_output - Predicted_test_output)./test_output ;
mean_test_error = sum(test_error)/length(test_error);