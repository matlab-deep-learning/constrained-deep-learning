function net = buildConvexCNN(inputSize, varargin)
% BUILDCONVEXCNN    Construct a fully input convex convolutional neural
%                   network.
%
%   NET = BUILDCONVEXCNN(INPUTSIZE, OUTPUTSIZE, FILTERSIZE, NUMFILTERS)
%   creates a fully input convex initialized dlnetwork object, NET.
%
%   The network includes either a sequenceInputLayer or an imageInputLayer,
%   depending on INPUTSIZE:
%
%   - If INPUTSIZE is a scalar, then the network has a sequenceInputLayer.
%   - If INPUTSIZE is a vector with two or three elements, then the
%     network has an imageInputLayer.
%
%   OUTPUTSIZE is an integer indicating the number of neurons in the output
%   fully connected layer.
%
%   FILTERSIZE defines the filter sizes for the convolutional layers. The
%   network will have as many convolutional layers as there are rows in
%   FILTERSIZE.
%
%   - If INPUTSIZE is a scalar, FILTERSIZE must be a column vector of
%     integers specifying the length of the filters for each
%     convolution1dLayer.
%   - If INPUTSIZE is a vector, FILTERSIZE is a matrix with two columns
%     specifying the height and width of the filters for each
%     convolution2dLayer. If FILTERSIZE is provided as a column vector, it
%     is assumed that the filters are square.
%
%   NUMFILTERS is a column vector of integers that specifies the number of
%   filters for each convolutional layer. It must have the same number of
%   rows as FILTERSIZE
%
%   NET = BUILDCONVEXCNN(__, NAME=VALUE) specifies additional options using
%   one or more name-value arguments.
%
%   These options and default values apply to fully-convex constrained
%   networks:
%
%   Stride                            - Stride for each convolutional
%                                       layer, specified as a matrix with
%                                       the same number of rows as
%                                       FILTERSIZE. If INPUTSIZE is a
%                                       scalar, Stride must be a column
%                                       vector specifying the stride
%                                       length. If INPUTSIZE is a vector,
%                                       Stride is a two-column matrix
%                                       specifying the stride width and
%                                       height. If INPUTSIZE is a scalar
%                                       and Stride is a column vector, a
%                                       square stride is assumed. The
%                                       default value is 1 for all layers.
%
%   DilationFactor                    - Dilation factor for each
%                                       convolutional layer, specified as
%                                       as matrix with the same number of
%                                       rows as FILTERSIZE. If INPUTSIZE is
%                                       a scalar, DilationFactor must be a
%                                       column vector specifying the
%                                       dilation length. If INPUTSIZE is a
%                                       vector, DilationFactor is a
%                                       two-column matrix specifying the
%                                       stride width and height. If
%                                       INPUTSIZE is a scalar and
%                                       DilationFactor is a column vector,
%                                       a square dilation factor is
%                                       assumed. The default value is 1 for
%                                       all layers.
%
%   Padding                           - Padding method for each
%                                       convolutional layer, specified as
%                                       "same" or "causal". Padding must be
%                                       a string array with the same number
%                                       of rows as FILTERSIZE. If INPUTSIZE
%                                       is a scalar, the default value is
%                                       "causal" for all layers. If
%                                       INPUTSIZE is a vector, the default
%                                       value is "same" for all layers.
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
    inputSize (1,:) {mustBeNonempty, mustBeReal, mustBeInteger, mustBePositive}
end

arguments(Repeating)
    varargin
end

switch numel(inputSize)
    case 1
        try
            net = conslearn.convex.cnn.buildSequenceFICCNN(inputSize, varargin{:});
        catch ME
            throwAsCaller(ME);
        end
    case {2, 3}
        try
            net = conslearn.convex.cnn.buildImageFICCNN(inputSize, varargin{:});
        catch ME
            throwAsCaller(ME);
        end
    otherwise
        error("Input size must be a vector with one, two, or three elements.");
end

end


