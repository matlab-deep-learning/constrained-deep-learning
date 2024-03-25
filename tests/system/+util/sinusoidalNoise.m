function noise = sinusoidalNoise(x)
%SINUSOIDALNOISE Additive sinusoidal noise

% Copyright 2024 The MathWorks, Inc.
noise = 0.1*sin(5*pi*x);
end