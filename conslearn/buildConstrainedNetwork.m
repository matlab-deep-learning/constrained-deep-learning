function net = buildConstrainedNetwork(constraint, inputSize, numHiddenUnits, options)
% BUILDCONSTRAINEDNETWORK    Construct a constrained multi-layer
%                            perceptron.
%
%   NET = BUILDCONSTRAINEDNETWORK(CONSTRAINT, INPUTSIZE, NUMHIDDENUNITS)
%   creates an initialized dlnetwork object, NET, which has the constraint
%   specified by CONSTRAINT, where CONSTRAINT is specified as:
%   "fully-convex", "partially-convex", "fully-monotonic",
%   "partially-monotonic", or "lipschitz".
%
%   The network includes either a featureInputLayer or an imageInputLayer,
%   depending on INPUTSIZE:
%
%   - If INPUTSIZE is a scalar, then the network has a featureInputLayer. -
%   If INPUTSIZE is a vector with three elements, then the network has an
%     imageInputLayer.
%
%   NUMHIDDENUNITS is a vector of integers that corresponds to the sizes
%   and number of fully connected layers in the network.
%
%   NET = BUILDCONSTRAINEDNETWORK(__, NAME=VALUE) specifies additional
%   options using one or more name-value arguments.
%
%   BUILDCONSTRAINEDNETWORK name-value arguments depend on the value of
%   CONSTRAINT.
%
%   These options and default values apply to convex constrained networks:
%
%   ConvexNonDecreasingActivation     - Convex, non-decreasing
%   ("fully-convex")                    activation functions.
%   ("partially-convex")                The options are "softplus" or
%   "relu".
%                                       The default is "softplus".
%   Activation                        - Network activation function.
%   ("partially-convex")                The options are "tanh", "relu" or
%                                       "fullsort". The default is "tanh".
%   ConvexChannelIdx                  - Channel indices for the inputs that
%   ("partially-convex")                carry convex dependency with the
%                                       output, specified as a vector of
%                                       positive integers. For image
%                                       inputs, the convex channel indices
%                                       correspond to the indices in the
%                                       flattened image input. The default
%                                       value is 1.
%
%   These options and default values apply to monotonic constrained
%   networks:
%
%   Activation                        - Network activation function.
%   ("fully-monotonic")                 The options are "tanh", "relu" or
%   ("partially-monotonic")             "fullsort".
%                                       The default is "fullsort".
%   ResidualScaling                   - The scale factor applied to the sum
%   ("fully-monotonic")                 of the inputs that carry monotonic
%   ("partially-monotonic")             dependency with the output.
%                                       The default value is 1.
%   MonotonicTrend                    - Monotonic trend of the output with
%   ("fully-monotonic")                 respect to increasing inputs,
%   ("partially-monotonic")             specified as either "increasing" or
%                                       "decreasing". The default is
%                                       "increasing".
%   pNorm                             - p-norm value for measuring
%   ("fully-monotonic")                 distance with respect to the
%   ("partially-monotonic")             Lipschitz continuity definition.
%                                       The default value is Inf.
%   MonotonicChannelIdx               - Channel indices for the inputs that
%   ("partially-monotonic")             carry monotonic dependency with the
%                                       output, specified as a vector of
%                                       positive integers. For image
%                                       inputs, the monotonic channel
%                                       indices correspond to the indices
%                                       in the flattened image input. The
%                                       default value is 1.
%
%   The following options and default values apply to Lipschitz constrained
%   networks:
%
%   Activation                        - Network activation function.
%                                       The options are "tanh", "relu" or
%                                       "fullsort". The default is
%                                       "fullsort".
%   UpperBoundLipschitzConstant       - Upper bound on the Lipschitz
%   constant
%                                       for the network, as a positive real
%                                       number. The default value is 1.
%   pNorm                             - p-norm value for measuring
%                                       distance with respect to the
%                                       Lipschitz continuity definition.
%                                       The default value is 1.
%
% [1] Amos, Brandon, et al. Input Convex Neural Networks. arXiv:1609.07152,
% arXiv, 14 June 2017. arXiv.org,
% https://doi.org/10.48550/arXiv.1609.07152. [2] Kitouni, Ouail, et al.
% Expressive Monotonic Neural Networks. arXiv:2307.07512, arXiv, 14 July
% 2023. arXiv.org, http://arxiv.org/abs/2307.07512.

%   Copyright 2024 The MathWorks, Inc.

arguments
    constraint {...
        mustBeTextScalar, ...
        mustBeMember(constraint,["fully-convex","partially-convex","fully-monotonic","partially-monotonic","lipschitz"])}
    inputSize (1,:) {mustBeInteger,mustBeReal,mustBePositive,...
        iValidateInputSize(inputSize)}
    numHiddenUnits (1,:) {mustBeInteger,mustBeReal,mustBePositive}
    % Convex
    options.ConvexNonDecreasingActivation {...
        mustBeTextScalar, ...
        mustBeMember(options.ConvexNonDecreasingActivation,["relu","softplus"]),...
        iValidateConstraintWithConvexNonDecreasingActivation(options.ConvexNonDecreasingActivation, constraint)}
    options.ConvexChannelIdx (1,:) {...
        iValidateConstraintWithConvexChannelIdx(options.ConvexChannelIdx, inputSize, constraint), ...
        mustBeNumeric,mustBePositive,mustBeInteger}
    % Lipschitz
    options.UpperBoundLipschitzConstant (1,1) {...
        iValidateConstraintWithUpperBoundLipschitzConstant(options.UpperBoundLipschitzConstant, constraint), ...
        mustBeNumeric,mustBePositive}
    % Monotonic
    options.MonotonicChannelIdx (1,:) {...
        iValidateConstraintWithMonotonicChannelIdx(options.MonotonicChannelIdx, inputSize, constraint), ...
        mustBeNumeric,mustBePositive,mustBeInteger}
    options.MonotonicTrend (1,1) {...
        mustBeTextScalar, ...
        iValidateConstraintWithMonotonicTrend(options.MonotonicTrend, constraint)}
    options.ResidualScaling (1,1) {...
        iValidateConstraintWithResidualScaling(options.ResidualScaling, constraint), ...
        mustBeNumeric,mustBePositive}
    % Lipschitz & Monotonic
    options.pNorm (1,1) {...
        iValidatePNorm(options.pNorm),...
        iValidateConstraintWithPNorm(options.pNorm, constraint)}
    % Convex, Lipschitz & Monotonic
    options.Activation {...
        mustBeTextScalar, ...
        mustBeMember(options.Activation,["relu","tanh","fullsort"]),...
        iValidateConstraintWithActivation(options.Activation, constraint)}
end
% Switch case to pass to builders
switch constraint
    case "fully-convex"
        % Set defaults
        if ~any(fields(options) == "ConvexNonDecreasingActivation")
            options.ConvexNonDecreasingActivation = "softplus";
        end
        net = conslearn.convex.buildFICNN(inputSize, numHiddenUnits, ...
            ConvexNonDecreasingActivation=options.ConvexNonDecreasingActivation);
    case "partially-convex"
        % Set defaults
        if ~any(fields(options) == "ConvexNonDecreasingActivation")
            options.ConvexNonDecreasingActivation = "softplus";
        end
        if ~any(fields(options) == "Activation")
            options.Activation = "tanh";
        end
        if ~any(fields(options) == "ConvexChannelIdx")
            options.ConvexChannelIdx = 1;
        end
        net = conslearn.convex.buildPICNN(inputSize, numHiddenUnits,...
            ConvexNonDecreasingActivation=options.ConvexNonDecreasingActivation,...
            Activation=options.Activation,...
            ConvexChannelIdx=options.ConvexChannelIdx);
    case "fully-monotonic"
        % Set defaults
        if ~any(fields(options) == "ResidualScaling")
            options.ResidualScaling = 1;
        end
        if ~any(fields(options) == "Activation")
            options.Activation = "fullsort";
        end
        if ~any(fields(options) == "MonotonicTrend")
            options.MonotonicTrend = "increasing";
        end
        if ~any(fields(options) == "pNorm")
            options.pNorm = Inf;
        end
        net = conslearn.monotonic.buildFMNN(inputSize, numHiddenUnits,...
            ResidualScaling=options.ResidualScaling,...
            Activation=options.Activation,...
            MonotonicTrend=options.MonotonicTrend,...
            pNorm=options.pNorm);
    case "partially-monotonic"
        % Set defaults
        if ~any(fields(options) == "ResidualScaling")
            options.ResidualScaling = 1;
        end
        if ~any(fields(options) == "Activation")
            options.Activation = "fullsort";
        end
        if ~any(fields(options) == "MonotonicTrend")
            options.MonotonicTrend = "increasing";
        end
        if ~any(fields(options) == "pNorm")
            options.pNorm = Inf;
        end
        if ~any(fields(options) == "MonotonicChannelIdx")
            options.MonotonicChannelIdx = 1;
        end
        net = conslearn.monotonic.buildPMNN(inputSize, numHiddenUnits,...
            ResidualScaling=options.ResidualScaling,...
            Activation=options.Activation,...
            MonotonicTrend=options.MonotonicTrend,...
            pNorm=options.pNorm,...
            MonotonicChannelIdx=options.MonotonicChannelIdx);
    case "lipschitz"
        % Set defaults
        if ~any(fields(options) == "UpperBoundLipschitzConstant")
            options.UpperBoundLipschitzConstant = 1;
        end
        if ~any(fields(options) == "Activation")
            options.Activation = "fullsort";
        end
        if ~any(fields(options) == "pNorm")
            options.pNorm = 1;
        end
        net = conslearn.lipschitz.buildLNN(inputSize, numHiddenUnits,...
            UpperBoundLipschitzConstant=options.UpperBoundLipschitzConstant,...
            Activation=options.Activation,...
            pNorm=options.pNorm);
end
end

function iValidateInputSize(inputSize)
if ~isequal(numel(inputSize),1) && ~isequal(numel(inputSize),3)
    error("The inputSize must be a scalar or a row vector with three elements.");
end
end

function iValidateConstraintWithUpperBoundLipschitzConstant(param, constraint)
if ~isequal(constraint,"lipschitz") && ~isempty(param)
    error("'UpperBoundLipschitzConstant' is not an option for constraint " + constraint);
end
end

function iValidateConstraintWithMonotonicChannelIdx(param, inputSize, constraint)
if ~isequal(constraint,"partially-monotonic") && ~isempty(param)
    error("'MonotonicChannelIdx' is not an option for constraint " + constraint);
end
if ~isempty(param) && any(param > prod(inputSize))
    error("'MonotonicChannelIdx' value is larger than the number input channels.");
end
end

function iValidateConstraintWithConvexChannelIdx(param, inputSize, constraint)
if ~isequal(constraint,"partially-convex") && ~isempty(param)
    error("'ConvexChannelIdx' is not an option for constraint " + constraint);
end
if ~isempty(param) && any(param > prod(inputSize))
    error("'ConvexChannelIdx' value is larger than the number input channels.");
end
if ~isempty(param) && isequal(numel(param),prod(inputSize))
    error("Number of convex channels specified by 'ConvexChannelIdx' is equal to the total number of inputs. For convexity in all inputs, specify the constraint as 'fully-convex'.");
end
if ~isempty(param) && ~isequal(numel(unique(param)),numel(param))
    error("'ConvexChannelIdx' must contain unique values.");
end
end

function iValidateConstraintWithResidualScaling(param, constraint)
if ( ~isequal(constraint, "fully-monotonic") && ~isequal(constraint,"partially-monotonic") ) && ~isempty(param)
    error("'ResidualScaling' is not an option for constraint " + constraint);
end
end

function iValidateConstraintWithMonotonicTrend(param, constraint)
if ( ~isequal(constraint, "fully-monotonic") && ~isequal(constraint,"partially-monotonic") ) && ~isempty(param)
    error("'MonotonicTrend' is not an option for constraint " + constraint);
end
end

function iValidateConstraintWithConvexNonDecreasingActivation(param, constraint)
if ( ~isequal(constraint, "fully-convex") && ~isequal(constraint,"partially-convex") ) && ~isempty(param)
    error("'ConvexNonDecreasingActivation' is not an option for constraint " + constraint);
end
end

function iValidateConstraintWithActivation(param, constraint)
if isequal(constraint, "fully-convex") && ~isempty(param)
    error("'Activation' is not an option for constraint " + constraint);
end
end

function iValidatePNorm(param)
if (~isequal(param,1) && ~isequal(param,2) && ~isequal(param,Inf)) && ~isempty(param)
    error("Invalid 'PNorm' value. Value must be 1, 2, or Inf.")
end
end

function iValidateConstraintWithPNorm(param, constraint)
if ( ~isequal(constraint, "fully-monotonic") && ~isequal(constraint, "partially-monotonic") && ~isequal(constraint, "lipschitz") ) && ~isempty(param)
    error("'PNorm' is not an option for constraint " + constraint);
end
end