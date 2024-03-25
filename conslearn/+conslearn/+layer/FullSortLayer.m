classdef FullSortLayer < nnet.layer.Layer & nnet.layer.Formattable & nnet.layer.Acceleratable
    % FullSortLayer    Sort in the channel space

    %   Copyright 2024 The MathWorks, Inc.

    methods
        function this = FullSortLayer(name)
            this.Name = name;
        end

        function Z = predict(~,X)
            cdim = finddim(X,'C');
            Z = sort(X,cdim);
        end
    end
end