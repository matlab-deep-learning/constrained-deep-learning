classdef TransformSequenceToFeatureLayer < nnet.layer.Layer & nnet.layer.Formattable & nnet.layer.Acceleratable
    %TransformSequenceToFeatureLayer    Transform a single channel sequence
    %input (1(C) x n(T)) into a two channel batch data (2(C) x n(B)).
    % - The first channel is first time step value of the sequence,
    % repeated for every batch value. 
    % - The second channel is the time index, normalized by some fixed
    % constant, TimeStepNormalization.

    %   Copyright 2024 The MathWorks, Inc.
    
    properties
        TimeStepNormalization
    end

    methods
        function this = TransformSequenceToFeatureLayer(timeStepNormalization)
            this.Name = "seq_channel";
            this.TimeStepNormalization = timeStepNormalization;
        end

        function Z = predict(this,X)
            % X is expect to be 'CT'
            % First make time steps batch observations
            Z = dlarray(X,'CB');           
            % Second, replicate t=0 at every batch slice
            Z = 0*Z + Z(:,1);
            % Third, add time steps as an extra channel
            Z(end+1,:) = (1:size(Z,2))'./this.TimeStepNormalization;
        end
    end
end