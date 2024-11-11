function net = buildImageFICCNN(inputSize, outputSize, filterSize, numFilters, options)
% BUILDIMAGEFICCNN     Construct a fully-input convex convolutional neural
%                      network for image inputs.
%
%   NET = BUILDIMAGEFICCNN(INPUTSIZE, OUTPUTSIZE, FILTERSIZE, NUMFILTERS)
%   creates a fully-input convex dlnetwork object, NET.
%
%   INPUTSIZE is a row vector of integers [h w c], where h, w, and c
%   correspond ot the height, width and number of channels respectively.
%
%   OUTPUTSIZE is an intenger indicating the number of neurons in the
%   output fully connected layer.
%
%   FILTERSIZE is matrix with two columns specifying the height and width
%   for each convolutional layer. The network will have as many
%   convolutional layers as there are rows in FILTERSIZE. If FILTERSIZE is
%   provided as a column vector, it is assumed that the filters are square.
%
%   NUMFILTERES is a column vector of integers that specifies the number of
%   filters for each convolutional layers. It must have the same number of
%   rows as FILTERSIZE.
%
%   NET = BUILDIMAGEFICCNN(__, NAME=VALUE) specifies additional options
%   using one or more name-value arguments.
%
%   Stride                            - Stride for each convolutional
%                                       layer, specified as a two-column
%                                       matrix where the first column is
%                                       the stride height, and the second
%                                       column is the stride width. If
%                                       Stride is specified as a column
%                                       vector, a square stride is assumed.
%                                       The default value is 1 for all
%                                       layers.
%
%   DilationFactor                    - Dilation factor for each
%                                       convolutional layer, specified as a
%                                       two-column matrix where the frist
%                                       column is the stride height and the
%                                       second column is the stride width.
%                                       If DilationFactor is a column
%                                       vector, a square dilation factor is
%                                       assumed. The default value is 1 for
%                                       all layers.
%
%   Padding                           - Padding method for each
%                                       convolutional layer specified as
%                                       "same" or "causal". Padding must be
%                                       a string array with the same number
%                                       of rows as FITLERSIZE. The default
%                                       value is "causal" for all layers.
%
%   PaddingValue                      - Padding for each convolutional
%                                       layer, specified as a column vector
%                                       with the same number of rows as
%                                       FILTERSIZE. The default value is 0
%                                       for all layers.
%
%   ConvexNonDecreasingActivation     - Convex non-decreasing activation
%                                       function, specified as "softplus"
%                                       or "relu". The default value is
%                                       "relu".

%   Copyright 2024 The MathWorks, Inc.

arguments
    inputSize (1,:) {mustBeNonempty, mustBeReal, mustBeInteger, mustBePositive, mustBeTwoOrThreeRowVector(inputSize, "inputSize")}
    outputSize (1,1) {mustBeReal, mustBeInteger, mustBePositive}
    filterSize {mustBeNonempty, mustBeReal, mustBeInteger, mustBePositive, mustBeOneOrTwoColumn(filterSize, "filterSize")}
    numFilters (:,1) {mustBeNonempty, mustBeReal, mustBeInteger, mustBePositive, mustBeEqualLength(filterSize, numFilters, "numFilters")}
    options.Stride {mustBeNonempty, mustBeReal, mustBeInteger, mustBePositive, mustBeOneOrTwoColumn(options.Stride, "Stride"), mustBeEqualLength(filterSize, options.Stride, "Stride")} = ones(numel(numFilters), 2)
    options.DilationFactor {mustBeNonempty, mustBeReal, mustBeInteger, mustBePositive, mustBeOneOrTwoColumn(options.DilationFactor, "DilationFactor"), mustBeEqualLength(filterSize, options.DilationFactor, "DilationFactor")} = ones(numel(numFilters), 2)
    options.Padding (:,1) {mustBeNonzeroLengthText, mustBeMember(options.Padding, "same"), mustBeEqualLength(filterSize, options.Padding, "Padding")} = repelem("same", numel(numFilters));
    options.PaddingValue (:,1) {mustBeNonempty, mustBeReal, mustBeEqualLength(filterSize, options.PaddingValue, "PaddingValue")} = zeros(numel(numFilters), 1);
    options.ConvexNonDecreasingActivation {mustBeNonzeroLengthText, mustBeTextScalar, mustBeMember(options.ConvexNonDecreasingActivation, ["relu", "softplus"])} = "relu"
end



% Select the activation function based on user input
switch options.ConvexNonDecreasingActivation
    case "relu"
        activationLayer = @(name) reluLayer(Name=name);
    case "softplus"
        activationLayer = @(name) softplusLayer(Name=name);
end

% Build the input layer
layers = [
    imageInputLayer(inputSize, Name="input", Normalization="none")
    ];

% Build the convolutional layers
for ii = 1:numel(numFilters)
    convLayerName = "conv2d_+_" + ii;
    activationLayerName = "cnd_" + ii;
    batchNormLayerName = "batchnorm_+_" + ii;

    convLayer = convolution2dLayer(filterSize(ii, :), numFilters(ii), ...
        Stride=options.Stride(ii, :), ...
        DilationFactor=options.DilationFactor(ii, :), ...
        Padding=options.Padding(ii), ...
        PaddingValue=options.PaddingValue(ii), ...
        Name=convLayerName);

    layers = [
        layers;
        convLayer;
        activationLayer(activationLayerName);
        batchNormalizationLayer(Name=batchNormLayerName)
        ]; %#ok<AGROW>
end

% Modify the name of the first convolutional layer to remove constraints
layers(2).Name = "conv2d_1";

% Add final pooling and fully connected layers
layers = [
    layers;
    globalAveragePooling2dLayer(Name="global_avg_pool");
    fullyConnectedLayer(outputSize, Name="fc_+_end")
    ];

% Initialize the dlnetwork
net = dlnetwork(layers);

% Make the network convex
net = conslearn.convex.makeNetworkConvex(net);

end

function mustBeTwoOrThreeRowVector(x, name)
if ~(isrow(x) && (numel(x) == 2 || numel(x) == 3))
    error("'%s' must be a row vector with two or three elements.", name);
end
end

function mustBeOneOrTwoColumn(x, name)
if ~(size(x, 2) == 1 || size(x, 2) == 2)
    error("'%s' must be an array with one or two columns.", name);
end
end

function mustBeEqualLength(filterSize, otherVar, otherVarName)
if ~isequal(size(filterSize, 1), size(otherVar, 1))
    error("'%s' must have the same number of rows as the filter size value.", otherVarName);
end
end
