function pnorm = applyInducedPNorm(W,p)
% APPLYINDUCEDPNORM    Applies the induced matrix p-norm, for p = 1, 2, Inf.

%   Copyright 2024 The MathWorks, Inc.

if isequal(p,1)
    pnorm = max(sum(abs(W),1));
elseif isequal(p,2)   
    pnorm = norm(extractdata(W),2);
    pnorm = dlarray(pnorm);
elseif isequal(p,Inf)
    pnorm = max(sum(abs(W),2));
else
    error("Only p-norms with p = 1, 2, Inf are valid.")
end
end