function net = trainConstrainedNetwork(constraint,net,mbq,trainingOptions)
% TRAINCONSTRAINEDNETWORK    Train a constrained neural network using
% adaptive momentum estimation (ADAM) solver.
%
%   NET = TRAINCONSTRAINEDNETWORK(CONSTRAINT, NET, MBQ) trains an
%   initialized dlnetwork object, NET, constructed to have the constraint
%    CONSTRAINT, specified as one of these options: "fully-convex",
%   "partially-convex", "fully-monotonic", "partially-monotonic", or
%   "lipschitz". The function preserves the constraint. The data, MBQ, is
%   specified as a minibatchqueue object.
%
%   NET = TRAINCONSTRAINEDNETWORK(__,NAME=VALUE) specifies additional
%   training options using one or more name-value arguments.
%
%   InitialLearnRate         - Initial learning rate for training. If the 
%                              learning rate is too low, training will 
%                              take a long time, but if it is too high,
%                              then the training is likely to get stuck at 
%                              a suboptimal result. 
%                              The default is 0.01.
%   MaxEpochs                - Maximum number of epochs for training. 
%                              The default is 30.
%   Decay                    - During training, drop the learning rate
%                              according to the expression, r/(1+n*x),
%                              where r is the InitialLearnRate value, x is
%                              the Decay value, and n is the number of 
%                              training iterations. A value of 0 corresponds
%                              to no drop in learn rate. 
%                              The default is 0.01.
%   LossMetric               - Metric to calculate loss at the end of each
%                              iteration, specified as: "mse", "mae", or
%                              "crossentropy". 
%                              The default is "mse".
%   L2Regularization         - Factor for L2 regularization (weight decay).
%                              The default is 0.
%   ValidationData           - Data to use for validation during training,
%                              specified as a minibatchqueue object.
%   ValidationFrequency      - Frequency of validation in number of
%                              iterations. The default is 50.
%   TrainingMonitor          - Flag to display the training progress monitor
%                              showing the training data loss. 
%                              The default is true.
%   TrainingMonitorLogScale  - Flag to display the training loss in log scale. 
%                              The default is true.
%   ShuffleMinibatches       - Flag to shuffle the minibatchqueue before every
%                              training epoch.
%                              The default is false.
%
%   TRAINCONSTRAINEDNETWORK name-value arguments that are valid when
%   CONSTRAINT is "fully-monotonic", "partially-monotonic":
%
%   pNorm                        - p-norm to measure distance with respect 
%                                  to the Lipschitz continuity definition.
%                                  The default value is Inf.
%
%   TRAINCONSTRAINEDNETWORK name-value arguments that are valid when
%   CONSTRAINT is "lipschitz":
%
%   UpperBoundLipschitzConstant  - Upper bound on the Lipschitz constant for 
%                                  the network, specified as a positive real
%                                  number. 
%                                  The default value is 1.
%   pNorm                        - p-norm to measure distance with respect 
%                                  to the Lipschitz continuity definition.
%                                  The default value is 1.

%   Copyright 2024 The MathWorks, Inc.

arguments
    constraint {...
        mustBeTextScalar, ...
        mustBeMember(constraint,["fully-convex","partially-convex","fully-monotonic","partially-monotonic","lipschitz"])}
    net (1,1) dlnetwork
    mbq (1,1) minibatchqueue
    % Options
    trainingOptions.MaxEpochs (1,1) {mustBeNumeric,mustBePositive,mustBeInteger} = 30
    trainingOptions.InitialLearnRate (1,1) {mustBeNumeric,mustBePositive} = 0.01
    trainingOptions.Decay (1,1) {mustBeNumeric,mustBePositive} = 0.01
    trainingOptions.LossMetric {...
        mustBeTextScalar, ...
        mustBeMember(trainingOptions.LossMetric,["mse","mae","crossentropy"])} = "mse";
    trainingOptions.L2Regularization (1,1) {mustBeNumeric, mustBeNonnegative} = 0
    trainingOptions.ValidationData minibatchqueue {mustBeScalarOrEmpty} = minibatchqueue.empty
    trainingOptions.ValidationFrequency (1,1) {mustBeNumeric, mustBePositive, mustBeInteger} = 50
    trainingOptions.TrainingMonitor (1,1) logical = true;
    trainingOptions.TrainingMonitorLogScale (1,1) logical = true;
    trainingOptions.ShuffleMinibatches (1,1) logical = false;
    % Lipschitz and Monotonic training options
    trainingOptions.pNorm (1,1)
    trainingOptions.UpperBoundLipschitzConstant (1,1) {mustBeNumeric,mustBePositive,mustBeFinite} = 1;
end

% Set up the training progress monitor
if trainingOptions.TrainingMonitor
    monitor = trainingProgressMonitor;

    % Track progress information 
    monitor.Info = ["LearningRate","Epoch","Iteration"];

    % Plot the training and validation metrics on the same plot
    monitor.Metrics = ["TrainingLoss", "ValidationLoss"];
    groupSubPlot(monitor, "Loss", ["TrainingLoss", "ValidationLoss"]);

    % Apply loss log scale
    if trainingOptions.TrainingMonitorLogScale
        yscale(monitor,"Loss","log");
    end

    % Specify the horizontal axis label for the training plot. 
    monitor.XLabel = "Iteration";

    % Start the monitor
    monitor.Status = "Running";
    stopButton = @() ~monitor.Stop;
else
    % Let training run without a monitor by setting stop to false
    stopButton = @() 1;
end

% Prepare the generic hyperparameters
maxEpochs = trainingOptions.MaxEpochs;
initialLearnRate = trainingOptions.InitialLearnRate;
decay = trainingOptions.Decay;
metric = trainingOptions.LossMetric;
shuffleMinibatches = trainingOptions.ShuffleMinibatches;
l2Regularization = trainingOptions.L2Regularization;
validationData = trainingOptions.ValidationData;
validationFrequency = trainingOptions.ValidationFrequency;

% Specify ADAM options
avgG = [];
avgSqG = [];

% Initialize training loop variables
epoch = 0;
iteration = 0;

% Setup proximal operator
% Set the default pNorm depending on constraint if unset by user.
if ~any(fields(trainingOptions) == "pNorm")
    if isequal(constraint,"fully-monotonic") || isequal(constraint,"partially-monotonic")
        trainingOptions.pNorm = Inf;
    elseif isequal(constraint,"lipschitz")
        trainingOptions.pNorm = 1;
    end
else
    iValidatePNorm(trainingOptions.pNorm);
end
proximalOp = iSetupProximalOperator(constraint,trainingOptions);

while epoch < maxEpochs && stopButton()
    epoch = epoch + 1;

    % Reset data.
    if shuffleMinibatches
        shuffle(mbq);
    else
        reset(mbq);
    end

    while hasdata(mbq) && stopButton()
        iteration = iteration + 1;

        % Read mini-batch of data.
        [X,T] = next(mbq);

        % Determine learning rate for time-based decay learning rate schedule.
        learnRate = initialLearnRate/(1 + decay*iteration);

        % Evaluate the model gradients, and loss using dlfeval and the
        % modelLoss function and update the network state.
        [lossTrain,gradients,state] = dlfeval(dlaccelerate(@iModelLoss),net,X,T,metric,l2Regularization);
        net.State = state;

        % Gradient Update
        [net,avgG,avgSqG] = adamupdate(net,gradients,avgG,avgSqG,epoch,learnRate);

        % Proximal Update
        net = proximalOp(net);

        % Update the training monitor
        if trainingOptions.TrainingMonitor
            updateInfo(monitor, ...
                LearningRate=learnRate, ...
                Epoch=string(epoch) + " of " + string(maxEpochs), ...
                Iteration=string(iteration));

            recordMetrics(monitor,iteration, ...
                TrainingLoss=lossTrain);
            
            monitor.Progress = 100*epoch/maxEpochs;
        end

        % Record validation loss, if requested
        if ~isempty(validationData)
            if (iteration == 1) || (mod(iteration, validationFrequency) == 0)

                % Reset the validation data
                if ~hasdata(validationData)
                    reset(validationData);
                end

                % Compute the validation loss
                [X, T] = next(validationData);
                lossValidation = iModelLoss(net, X, T, metric, l2Regularization);

                % Update the training monitor
                if trainingOptions.TrainingMonitor
                    recordMetrics(monitor,iteration, ...
                        ValidationLoss=lossValidation);
                end
            end
        end
    end
end

% Update the training monitor status
if trainingOptions.TrainingMonitor
    if monitor.Stop == 1
        monitor.Status = "Training stopped";
    else
        monitor.Status = "Training complete";
    end
end

end

%% Helpers
function [loss,gradients,state] = iModelLoss(net,X,T,metric,l2Regularization)

% Make a forward pass
[Y, state] = forward(net,X);

% Compute the loss
switch metric
    case "mse"
        loss = mse(Y,T);
    case "mae"
        loss = mean(abs(Y-T), 'all');
    case "crossentropy"
        loss = crossentropy(softmax(Y),T);
end

if nargout > 1
    % Compute the gradient of the loss with respect to the learnables
    gradients = dlgradient(loss,net.Learnables);

    % Apply L2 regularization
    idxWeights = net.Learnables.Parameter == "Weights";
    gradients(idxWeights,:) = dlupdate(@(g,w) g + l2Regularization*w, gradients(idxWeights, :), net.Learnables(idxWeights, :));
end
end

function proximalOp = iSetupProximalOperator(constraint,trainingOptions)
switch constraint
    case "fully-convex"
        proximalOp = @(net) conslearn.convex.makeNetworkConvex(net);
    case "partially-convex"
        proximalOp = @(net) conslearn.convex.makeNetworkConvex(net);
    case "fully-monotonic"
        pNorm = trainingOptions.pNorm;
        proximalOp = @(net) conslearn.monotonic.makeNetworkMonotonic(net,pNorm);
    case "partially-monotonic"
        pNorm = trainingOptions.pNorm;
        proximalOp = @(net) conslearn.monotonic.makeNetworkMonotonic(net,pNorm);
    case "lipschitz"
        pNorm = trainingOptions.pNorm;
        lipschitzUpperBound = trainingOptions.UpperBoundLipschitzConstant;
        proximalOp = @(net) conslearn.lipschitz.makeNetworkLipschitz(net,pNorm,lipschitzUpperBound);
end
end

function iValidatePNorm(param)
if (~isequal(param,1) && ~isequal(param,2) && ~isequal(param,Inf)) && ~isempty(param)
error("Invalid 'PNorm' value. Value must be 1, 2, or Inf.")
end
end