function [netMin,netMax,netPred] = computeICNNBoundsOverNDGrid(net,Vn,numOutputs)
% COMPUTEICNNBOUNDSOVERNDGRID    Compute the upper and lower bounds of a
% convex neural network over a hypercubic grid.
%
%   [NETMIN,NETMAX] = COMPUTEICNNBOUNDSOVERNDGRID(NET, V, NUMOUTPUTS) 
%   computes the maximum and minimum of NET on the hypercubic
%   regions specified by V. Where the minimum is on the hypercubic
%   interior, or cannot be determined by gradient evaluations at the
%   hypercubic vertices, NETMIN has NaN elements.
%
%   [~,~,NETPRED] = COMPUTEICNNBOUNDSOVERNDGRID(NET, V, NUMOUTPUTS) compute
%   the predictions of the convex network, NET, where NET is a dlnetwork
%   object with NUMOUTPUTS channel outputs, on the vertices of the
%   hypercubic grid, V, where V is a cell array containing the outputs of
%   the NDGRID function for 1, 2 or 3 dimensions. NETPRED is the same shape
%   as an element of V, i.e., an NDGRID output shape.

%   Copyright 2024 The MathWorks, Inc.

arguments
    net (1,1) dlnetwork
    Vn (:,1) cell
    numOutputs (1,1) {mustBeInteger}
end
% Dimension of input domain
numInputs = net.Layers(1).InputSize;

% Prepare 
netMin = cell(numOutputs,1);
netMax = cell(numOutputs,1);
netPred = cell(numOutputs,1);

% Loop over each network output. This is required since the derivatives are
% taken with respect to each network output, summed in batch.
accFun = dlaccelerate(@computeFunctionAndDerivativeForScalarOutput);
for output = 1:numOutputs
    %% Max
    % Computing the max means looking at the largest function value at the
    % vertex of a hypercube.

    % Step 1 - Convert the gridded coordinates to CB array for consumption by
    % network
    X = [];
    for ii = 1:numInputs
        X = [X Vn{ii}(:)];
    end
    X = dlarray(X','CB');

    % Step 2 - Evaluate the network at the grid points and reshape to grid
    [Z,dZ] = dlfeval(accFun,net,X,output);
    Z = reshape(Z,size(Vn{1})); % reshape to ndgrid

    % Step 3 - Take max for each hypercube using maxpool operation. This gives
    % the network max on each hypercube.
    fMax = maxpool(Z,2*ones(1,numInputs),DataFormat=repelem('S',numInputs));

    % Computing the min involves looking at the directional derivatives at
    % the hypercubic vertices.

    % Step 1 - reshape derivatives. Each cell corresponds to a directional
    % derivative, i.e., ii=1 corresponds to the derivatives in the x1
    % direction.
    ddZ = cell(numInputs,1);
    for ii = 1:numInputs
        ddZ{ii} = reshape(dZ(ii,:),size(Vn{ii}));
    end

    % Step 2 - create the conv weights and bias. These will take the
    % correct directional derivative so that dZ points to the interior of
    % the hypercube. Each corner of the hypercube is computed as a separate
    % channel.
    if numInputs > 1
        % Create a hypercube in more than 1D
        twoCube = ones(2*ones(1,numInputs));
    else
        % Create an interval in 1D
        twoCube = [1; 1];
    end
    twoCube(2,:) = -1;
    weights = cell(numInputs,1);
    for ii = 1:numInputs
        kernel = shiftdim(twoCube,numInputs+1-ii);
        weights{ii} = diag(kernel(:));
        weights{ii} = reshape(weights{ii},[2*ones(1,numInputs),1,numel(kernel)]);
    end
    bias = zeros(1,numel(kernel));

    % Step 3 - compute the directional derivatives at each vertex
    for ii = 1:numInputs
        ddZ{ii} = dlconv(dlarray(ddZ{ii},[repelem('S',numInputs),'C']),weights{ii},bias);
    end
    % ddZ{ii} denotes the directional derivative with respect to the i-th
    % input. The format of ddZ{ii} is % m-1(S) x m-1(S) x ... x m-1(S) x
    % 2^n(C) where ii denotes a direction. The C dimensions are each corner
    % of the hypercube, traversing in a column-wise indexing manner. The
    % S...S dimensions are the grid set. For example element (1,1,3) would
    % be the 3rd corner on the square at position 1,1. (2,1,2) would be the
    % 2nd corner on the square at position 2,1.

    % Step 4 - Compute the signature of the derivatives at the corners and look
    % for any signatures (+,...,+).
    numVertices = 2^numInputs;
    S = cat(numInputs+2,ddZ{:});
    positiveInwardGradients = false(size(ddZ{1}));
    idx = repmat({':'},1,numInputs+2);
    for ii = 1:numVertices
        idx{numInputs+1} = ii;
        positiveInwardGradients(idx{1:end-1}) = all(S(idx{:}) >= 0,numInputs+2);
    end

    % Step 5 - For positiveInwardGradients values of 1, this corresponds to
    % corners that have the minimum function value. Leave fMin as NaN if
    % there is no positive gradient corner. This indicates the function
    % minimum lies within or on the edge of the hypercube.
    fMin = NaN(size(fMax));

    % Need to embed the positiveInwardGradients array into an array, 1
    % larger in each dimension, so that you can index into Z, the network
    % prediction at the vertices. This code pads the
    % positiveInwardGradients correctly so as to make sure the correct
    % vertices in Z are accessed. 
    % 
    % For instance, for a 2x2 grid, positiveInwardsGradients is 1x1x4 and Z
    % is 2x2. positiveInwardGradients(1,1,1) is the upper left corner, so
    % need to index into Z([1,0;0,0]). positiveInwardGradients(1,1,4) is
    % the bottom right corner, so need to index into Z([0,0;0,1]). Hence,
    % need to embed positiveInwardGradients correctly to index into Z.
    idx = repmat({':'},1,numInputs+1);
    for ii = 1:numVertices
        idx{numInputs+1} = ii;
        embedInwardGradients = padarray(positiveInwardGradients(idx{:}),ones(1,numInputs),0);
        sz = size(embedInwardGradients);
        xEmbed = iRowAndColumnToRemove(numInputs,sz,ii);
        for jj = 1:numel(xEmbed)
            idxCrop = idx(1:end-1);
            idxCrop{jj} = xEmbed{jj};
            embedInwardGradients(idxCrop{:}) = [];
        end
        % Assign the minimum values (if any at the vertex) to fMin
        fMin(positiveInwardGradients(idx{:})) = Z(embedInwardGradients);
    end

    % Assign to netPred, netMax, netMin
    netPred{output} = gather(extractdata(Z));
    netMax{output} = gather(extractdata(fMax));
    netMin{output} = gather(fMin);
end
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

function xEmbed = iRowAndColumnToRemove(n,sz,k)
xEmbed = cell(n,1);
if n > 1
    % Need to use ind2sub as nD-array
    [xEmbed{:}] = ind2sub(2*ones(1,n),k);
else
    % Use index direction as 1D-array
    [xEmbed{:}] = k;
end
for ii = 1:numel(xEmbed)
    if isequal(xEmbed{ii},2)
        xEmbed{ii} = sz(ii);
    end
end
end