function [chargingNet,dischargingNet,chargingDiffNet,dischargingDiffNet] = trainUnconstrainedNetworks(XTrainCharging,YTrainCharging,XTrainDischarging,YTrainDischarging,XTrainChargingDifference,YTrainChargingDifference,XTrainDischargingDifference,YTrainDischargingDifference, chargingOptions, dischargingOptions)

% Don't plot training progress

chargingOptions.Plots = "none";
dischargingOptions.Plots = "none";

% Train unconstrained RNN

numFeatures = size(XTrainCharging{1},2);
numHiddenUnits = 32;
numResponses = size(YTrainCharging{1},2);

layers = [sequenceInputLayer(numFeatures,"Normalization","rescale-zero-one")
    lstmLayer(numHiddenUnits)
    dropoutLayer(0.2)
    lstmLayer(numHiddenUnits/2)
    dropoutLayer(0.2)
    fullyConnectedLayer(numResponses)
    sigmoidLayer];


chargingNet = trainnet(XTrainCharging,YTrainCharging,layers,"mse",chargingOptions);

dischargingNet = trainnet(XTrainDischarging,YTrainDischarging,layers,"mse",dischargingOptions);

% Train network on differences

layersDiff = [sequenceInputLayer(numFeatures,"Normalization","rescale-zero-one")
    lstmLayer(numHiddenUnits)
    dropoutLayer(0.2)
    lstmLayer(numHiddenUnits/2)
    dropoutLayer(0.2)
    fullyConnectedLayer(numResponses)];

chargingDiffNet = trainnet(XTrainChargingDifference,YTrainChargingDifference,layersDiff,"mse",chargingOptions);

dischargingDiffNet = trainnet(XTrainDischargingDifference,YTrainDischargingDifference,layersDiff,"mse",dischargingOptions);

end