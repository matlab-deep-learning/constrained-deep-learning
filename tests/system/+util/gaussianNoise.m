function noise = gaussianNoise(x)
%GAUSSIANNOISE Additive White Gaussian Noise

% Copyright 2024 The MathWorks, Inc.
noise = 0.1*randn(size(x,1),1);
end