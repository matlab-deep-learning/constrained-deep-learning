function w = makeParametersConvex(w)
% MAKEPARAMETERSCONVEX    Apply positivity constraint to weights.

%   Copyright 2024 The MathWorks, Inc.

w = relu(w);
end