function refinedNetMin = refineICNNLowerBoundsOverNDGrid(net,Vn,netMin)
% REFINEICNNLOWERBOUNDSOVERNDGRID    Refines the lower bounds of a
% convex neural network over a hypercubic grid using convex optimization.
%
%   REFINEDNETMIN = REFINEICNNLOWERBOUNDSOVERNDGRID(NET, V, NETMIN) refines
%   the lower bounds of the convex network, NETMIN, returned within
%   COMPUTEICNNBOUNDSOVERNDGRID by computing convex optimization for the
%   minimum values in hypercubic regions where the minimum could not be
%   determined by gradient analysis of the vertices alone. The function
%   uses the Optimization Toolbox routine fmincon to make such refinements.

%   Copyright 2024 The MathWorks, Inc.

arguments
    net (1,1) dlnetwork
    Vn (:,1) cell
    netMin (:,1) cell
end
% Dimension of input domain
numInputs = net.Layers(1).InputSize;
numOutputs = numel(netMin);

VnMax = cell(numInputs,1);
VnMin = cell(numInputs,1);

% Refined minima
refinedNetMin = netMin;
for output = 1:numOutputs
    for ii = 1:numInputs
        VnMax{ii} = maxpool(dlarray(Vn{ii}),2*ones(1,numInputs),...
            DataFormat=repelem('S',numInputs));
        VnMin{ii} = -maxpool(dlarray(-Vn{ii}),2*ones(1,numInputs),...
            DataFormat=repelem('S',numInputs));
    end
    reqOptimSolve = isnan(netMin{output});
    uLower = [];
    uUpper = [];
    for ii = 1:numInputs
        uLower = [uLower extractdata(VnMin{ii}(reqOptimSolve))]; %#ok<AGROW>
        uUpper = [uUpper extractdata(VnMax{ii}(reqOptimSolve))]; %#ok<AGROW>
    end

    options = optimoptions('fmincon',...
        Display='off',...
        Algorithm='interior-point',...
        SpecifyObjectiveGradient=true);
    objectiveFun = @(X) objectiveFcn(net,X,output);

    % Number of optim solves
    fval = zeros(nnz(reqOptimSolve),1);
    for kk = 1:nnz(reqOptimSolve)
        lb = uLower(kk,:);
        ub = uUpper(kk,:);
        % Initial guess within the hypercube
        x0 = mean([lb;ub],1);
        [~, fval(kk)] = fmincon(objectiveFun, x0, [], [], [], [], lb, ub, [], options);
    end

    refinedNetMin{output}(reqOptimSolve) = fval;
end
end

function [Z,dZ] = objectiveFcn(net,X,output)
X = dlarray(X,'BC');
accFun = dlaccelerate(@computeFunctionAndDerivativeForScalarOutput);
[Z,dZ] = dlfeval(accFun,net,X,output);
% Return double arrays on host
dZ = gather(double(extractdata(dZ)));
Z = gather(double(extractdata(Z)));
end

function [Z,dZ] = computeFunctionAndDerivativeForScalarOutput(net,X,output)
% Evaluate f
Z = predict(net,X);
% Evaluate df/dx
bdim = finddim(Z,'B');
% Since Z_i depends only on X_i, the derivative of d/dX_i sum(Z_i) = d/dX_i Z_i
dZ = dlgradient(sum(Z(output,:),bdim),X);
Z = Z(output,:);
end