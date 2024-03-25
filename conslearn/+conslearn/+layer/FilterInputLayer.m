classdef FilterInputLayer < nnet.layer.Layer & nnet.layer.Formattable & nnet.layer.Acceleratable
    % FilterInputLayer    Remove a subset of channel inputs. This layer
    % assumes that the first dimension is the channel dimension and the
    % data is in 'CB' format.

    %   Copyright 2024 The MathWorks, Inc.

    properties
        PermittedChannels
    end

    methods
        function this = FilterInputLayer(permittedChannels,name)
            this.Name = name;
            this.PermittedChannels = permittedChannels;
        end

        function Z = predict(this,X)
            Z = X(this.PermittedChannels,:);
        end
    end
end