function x = forwardEuler(net,tspan,y0)
% FORWARDEULER

%   Copyright 2024 The MathWorks, Inc.

arguments
    net dlnetwork
    tspan (1,:) {mustBeNumeric,mustBeReal,iValidateTspan}
    y0     {mustBeA(y0,'dlarray'),mustBeReal}
end
% dt set to min(diff(tspan)) by default.
dt = min(diff(tspan));

% Create odefun for network
numIn = numel(net.InputNames);
theta = net;
if numIn>2
    error("Expected network to have at most 2 inputs");
elseif numIn==2
    net = @iExpandTAndPredict;
else
    net = @(t,y,net) net.predict(y);
end

% Create fixed time steps to run Euler method over.
t = tspan(1):dt:(tspan(end)+dt);
xt = cell(numel(t),1);
xt{1} = y0;
for i = 2:numel(t)
    xt{i} = xt{i-1} + dt*net(t(i-1),xt{i-1},theta);
end
xt = cat(ndims(y0)+1,xt{:});
if ~isempty(dims(y0))
    xt = dlarray(xt,strcat(dims(y0),'T'));
end

% Now linearly interpolate to evaluate at the given tspan.
xt = stripdims(xt);
xt = permute(xt,[ndims(xt),1:(ndims(xt)-1)]);
x = interp1(t,xt,tspan(2:end).');

% Fix up output shape and dimensions
x = permute(x,[2:ndims(x),1]);

% potentially trailing singletons got dropped so recompute format
originalDims = dims(y0);
if ~isempty(dims(y0))
    if numel(tspan)==2
        newDims = originalDims;
    else
        newDims = [originalDims(1:ndims(x)-1),'T',originalDims(ndims(x):end)];
    end
    x = dlarray(x,newDims);
end
end

function iValidateTspan(tspan)
isDlarray = isdlarray(tspan);
if isDlarray && ~isempty(dims(tspan))
    error("Interval of integration must be a numeric array or an unformatted dlarray.");
elseif isDlarray
    tspan = extractdata(tspan);
end
if ~(numel(tspan) >= 2 && issorted(tspan,'strictmonotonic'))
    error("Time interval must be a strictly increasing or strictly decreasing row vector of at least two elements.");
end
end

function y = iExpandTAndPredict(t,y,theta)
% Expand t over batch dimension and cast to formatted dlarray.
bdim = finddim(y,'B');
if isempty(bdim)
    batchSize = 1;
else
    batchSize = size(y,bdim);
end
t = repmat(t,1,batchSize);
t = dlarray(t,"CB");

% Call net.predict
y = theta.predict(t,y);
end