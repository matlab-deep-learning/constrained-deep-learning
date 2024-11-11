function net = buildSequenceFICCNN(inputSize, outputSize, filterSize, numFilters, options)
% BUILDSEQUENCEFICCNN     Construct a fully-input convex convolutional
%                         neural network for sequence inputs.
%
%   NET = BUILDSEQUENCEFICCNN(INPUTSIZE, OUTPUTSIZE, FILTERSIZE,
%   NUMFILTERS) creates a fully-input convex dlnetwork object, NET.
%
%   INPUTSIZE is a integer indicating the number of features.
%
%   OUTPUTSIZE is an intenger indicating the number of neurons in the
%   output fully connected layer.
%
%   FILTERSIZE is column vector of integer filter sizes. The network will
%   have as many convolutional layers as there are rows in FILTERSIZE.
%
%   NUMFILTERES is a column vector of integers that specifies the number of
%   filters for each convolutional layers. It must have the same number of
%   rows as FILTERSIZE.
%
%   NET = BUILDSEQUENCEFICCNN(__, NAME=VALUE) specifies additional options
%   using one or more name-value arguments.
%
%   Stride                            - Stride for each convolutional
%                                       layer, specified as a column vector
%                                       of integers with the same number of
%                                       rows as FILTERSIZE. The default
%                                       value is 1 for all layers.
%
%   DilationFactor                    - Dilation factor for each
%                                       convolutional layer, specified as a
%                                       column vector with the same number
%                                       of rows as FILTERSIZE. The default
%                                       value is 1 for all layers.
%
%   Padding                           - Padding method for each
%                                       convolutional layer specified as
%                                       "same" or "causal". Padding must be
%                                       a a string array with the same
%                                       number of rows as FITLERSIZE. The
%                                       default value is "causal" for all
%                                       layers.
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
    inputSize (1,1) {mustBeReal, mustBeInteger, mustBePositive}
    outputSize (1,1) {mustBeReal, mustBeInteger, mustBePositive}
    filterSize (:,1) {mustBeNonempty, mustBeReal, mustBeInteger, mustBePositive}
    numFilters (:,1) {mustBeNonempty, mustBeReal, mustBeInteger, mustBePositive, mustBeEqualLength(filterSize, numFilters, "numFilters")}
    options.Stride (:,1) {mustBeNonempty, mustBeReal, mustBeInteger, mustBePositive, mustBeEqualLength(filterSize, options.Stride, "Stride")} = ones(numel(filterSize), 1)
    options.DilationFactor (:,1) {mustBeNonempty, mustBeReal, mustBeInteger, mustBePositive, mustBeEqualLength(filterSize, options.DilationFactor, "DilationFactor")} = ones(numel(filterSize), 1)
    options.Padding (:,1) {mustBeNonzeroLengthText, mustBeMember(options.Padding, ["same", "causal"]), mustBeEqualLength(filterSize, options.Padding, "Padding")} = repelem("causal", numel(filterSize))
    options.PaddingValue (:,1) {mustBeNonempty, mustBeReal, mustBeEqualLength(filterSize, options.PaddingValue, "PaddingValue")} = zeros(numel(filterSize), 1)
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
    sequenceInputLayer(inputSize, Name="input", Normalization="none")
    ];

% Build the convolutional layers
for ii = 1:numel(numFilters)
    convLayerName = "conv1d_+_" + ii;
    activationLayerName = "cnd_" + ii;
    batchNormLayerName = "batchnorm_+_" + ii;

    convLayer = convolution1dLayer(filterSize(ii), numFilters(ii), ...
        Stride=options.Stride(ii), ...
        DilationFactor=options.DilationFactor(ii), ...
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
layers(2).Name = "conv1d_1";

% Add final pooling and fully connected layers
layers = [
    layers;
    globalAveragePooling1dLayer(Name="global_avg_pool");
    fullyConnectedLayer(outputSize, Name="fc_+_end")
    ];

% Initialize the dlnetwork
net = dlnetwork(layers);

% Make the network convex
net = conslearn.convex.makeNetworkConvex(net);

end

function mustBeEqualLength(filterSize, otherVar, otherVarName)
if ~isequal(size(filterSize, 1), size(otherVar, 1))
    error("'%s' must have the same number of rows as the filter size value.", otherVarName);
end
end
