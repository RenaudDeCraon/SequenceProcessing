package SequenceProcessing.Classification;

import ComputationalGraph.*;
import ComputationalGraph.Function.*;
import ComputationalGraph.Node.*;
import Math.Tensor;
import SequenceProcessing.Functions.*;
import SequenceProcessing.Parameters.RecurrentNeuralNetworkParameter;

import java.io.Serializable;
import java.util.*;

public class LongShortTermMemoryModel extends RecurrentNeuralNetworkModel implements Serializable {

    public LongShortTermMemoryModel(int wordEmbeddingLength) {
        super(wordEmbeddingLength);
        this.switches = new ArrayList<>();
    }

    @Override
    public void train(ArrayList<Tensor> trainSet, NeuralNetworkParameter parameters) {
        Random random = new Random(parameters.getSeed());
        int timeStep = findTimeStep(trainSet);
        ArrayList<ComputationalNode> weights = new ArrayList<>();
        ArrayList<ComputationalNode> recurrentWeights = new ArrayList<>();
        int currentLength = wordEmbeddingLength + 1;
        for (int i = 0; i < ((RecurrentNeuralNetworkParameter) parameters).size(); i++) {
            for (int j = 0; j < 4; j++) {
                weights.add(new MultiplicationNode(true, false, new Tensor(parameters.getInitialization().initialize(currentLength, ((RecurrentNeuralNetworkParameter) parameters).getHiddenLayer(i), random), new int[]{currentLength, ((RecurrentNeuralNetworkParameter) parameters).getHiddenLayer(i)}), false));
                recurrentWeights.add(new MultiplicationNode(true, false, new Tensor(parameters.getInitialization().initialize(((RecurrentNeuralNetworkParameter) parameters).getHiddenLayer(i), ((RecurrentNeuralNetworkParameter) parameters).getHiddenLayer(i), random), new int[]{((RecurrentNeuralNetworkParameter) parameters).getHiddenLayer(i), ((RecurrentNeuralNetworkParameter) parameters).getHiddenLayer(i)}), false));
            }
            currentLength = ((RecurrentNeuralNetworkParameter) parameters).getHiddenLayer(i) + 1;
        }
        weights.add(new MultiplicationNode(true, false, new Tensor(parameters.getInitialization().initialize(currentLength, ((RecurrentNeuralNetworkParameter) parameters).getClassLabelSize(), random), new int[]{currentLength, ((RecurrentNeuralNetworkParameter) parameters).getClassLabelSize()}), false));
        ArrayList<ComputationalNode> currentOldLayers = new ArrayList<>();
        ArrayList<ComputationalNode> currentOldCValues = new ArrayList<>();
        ArrayList<ComputationalNode> outputNodes = new ArrayList<>();
        for (int k = 0; k < timeStep; k++) {
            this.switches.add(new Switch());
            ArrayList<ComputationalNode> newOldLayers = new ArrayList<>();
            ArrayList<ComputationalNode> newOldCValues = new ArrayList<>();
            ComputationalNode input = new MultiplicationNode(false, true, false);
            inputNodes.add(input);
            ComputationalNode current = input;
            for (int i = 0; i < weights.size() - 1; i += 4) {
                ComputationalNode aw;
                ComputationalNode aFunction;
                ComputationalNode ct;
                if (!currentOldLayers.isEmpty()) {
                    aw = this.addEdge(current, weights.get(i), false);
                    ComputationalNode oWithoutBias = this.addEdge(currentOldLayers.get(i / 4), new RemoveBias(), false);
                    ComputationalNode ou = this.addEdge(oWithoutBias, recurrentWeights.get(i), false);
                    ComputationalNode awOu = this.addAdditionEdge(aw, ou, false);
                    ComputationalNode it = this.addEdge(awOu, ((RecurrentNeuralNetworkParameter) parameters).getActivationFunction(i), false);
                    aw = this.addEdge(current, weights.get(i + 1), false);
                    ou = this.addEdge(oWithoutBias, recurrentWeights.get(i + 1), false);
                    awOu = this.addAdditionEdge(aw, ou, false);
                    ComputationalNode ft = this.addEdge(awOu, ((RecurrentNeuralNetworkParameter) parameters).getActivationFunction(i + 1), false);
                    aw = this.addEdge(current, weights.get(i + 2), false);
                    ou = this.addEdge(oWithoutBias, recurrentWeights.get(i + 2), false);
                    awOu = this.addAdditionEdge(aw, ou, false);
                    ComputationalNode ot = this.addEdge(awOu, ((RecurrentNeuralNetworkParameter) parameters).getActivationFunction(i + 2), false);
                    aw = this.addEdge(current, weights.get(i + 3), false);
                    ou = this.addEdge(oWithoutBias, recurrentWeights.get(i + 3), false);
                    awOu = this.addAdditionEdge(aw, ou, false);
                    ComputationalNode cTemp = this.addEdge(awOu, new Tanh(), false);
                    ComputationalNode ftCt1 = this.addEdge(ft, currentOldCValues.get(i / 4), false, true);
                    ComputationalNode itCtTemp = this.addEdge(it, cTemp, false, true);
                    ComputationalNode cmb = this.addAdditionEdge(ftCt1, itCtTemp, false);
                    ct = this.addEdge(cmb, ((RecurrentNeuralNetworkParameter) parameters).getActivationFunction(i + 3), false);
                    ComputationalNode tanhCt = this.addEdge(ct, new Tanh(), false);
                    aFunction = this.addEdge(tanhCt, ot, true, true);
                } else {
                    aw = this.addEdge(current, weights.get(i), false);
                    ComputationalNode it = this.addEdge(aw, ((RecurrentNeuralNetworkParameter) parameters).getActivationFunction(i), false);
                    aw = this.addEdge(current, weights.get(i + 1), false);
                    ComputationalNode ot = this.addEdge(aw, ((RecurrentNeuralNetworkParameter) parameters).getActivationFunction(i + 2), false);
                    aw = this.addEdge(current, weights.get(i + 3), false);
                    ComputationalNode cTemp = this.addEdge(aw, new Tanh(), false);
                    ComputationalNode itCTemp = this.addEdge(it, cTemp, false, true);
                    ct = this.addEdge(itCTemp, ((RecurrentNeuralNetworkParameter) parameters).getActivationFunction(i + 3), false);
                    ComputationalNode tanhCt = this.addEdge(ct, new Tanh(), false);
                    aFunction = this.addEdge(tanhCt, ot, true, true);
                }
                current = aFunction;
                newOldLayers.add(aFunction);
                newOldCValues.add(ct);
            }
            currentOldLayers = newOldLayers;
            currentOldCValues = newOldCValues;
            ComputationalNode node = this.addEdge(current, weights.get(weights.size() - 1), false);
            outputNodes.add(this.addEdge(node, switches.get(k), false));
        }
        ConcatenatedNode concatenatedNode = (ConcatenatedNode) this.concatEdges(outputNodes, 0);
        this.addEdge(concatenatedNode, new Softmax(), false);
        train(trainSet, parameters, random);
    }
}