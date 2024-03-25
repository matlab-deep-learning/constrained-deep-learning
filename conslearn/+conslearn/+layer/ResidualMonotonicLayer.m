classdef ResidualMonotonicLayer < nnet.layer.Layer & nnet.layer.Formattable & nnet.layer.Acceleratable
    % ResidualMonotonicLayer    Sum and scale a subset of channels

    %   Copyright 2024 The MathWorks, Inc.

    properties        
        MonotonicChannels
        ResidualScaling
    end

    methods
        function this = ResidualMonotonicLayer(monotonicChannels,lipschitzConstant)
            this.MonotonicChannels = monotonicChannels;
            this.ResidualScaling = lipschitzConstant;
            this.Name = 'res_mono';
        end

        function Z = predict(this,X)
            % Layer expects 'CB' dlarray
            Z = sum(X(this.MonotonicChannels,:),1);
            Z = this.ResidualScaling*Z;
        end
    end
end