package SequenceProcessing.Classification;

import ComputationalGraph.Function.Negation;
import ComputationalGraph.Function.Softmax;
import ComputationalGraph.Function.Tanh;
import ComputationalGraph.NeuralNetworkParameter;
import ComputationalGraph.Node.ComputationalNode;
import ComputationalGraph.Node.ConcatenatedNode;
import ComputationalGraph.Node.MultiplicationNode;
import SequenceProcessing.Functions.AdditionByConstant;
import SequenceProcessing.Functions.RemoveBias;
import SequenceProcessing.Functions.Switch;
import SequenceProcessing.Parameters.RecurrentNeuralNetworkParameter;
import Math.Tensor;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;

public class GatedRecurrentUnitModel extends RecurrentNeuralNetworkModel implements Serializable {

    public GatedRecurrentUnitModel(int wordEmbeddingLength) {
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
            for (int j = 0; j < 3; j++) {
                weights.add(new MultiplicationNode(true, false, new Tensor(parameters.getInitialization().initialize(currentLength, ((RecurrentNeuralNetworkParameter) parameters).getHiddenLayer(i), random), new int[]{currentLength, ((RecurrentNeuralNetworkParameter) parameters).getHiddenLayer(i)}), false));
                recurrentWeights.add(new MultiplicationNode(true, false, new Tensor(parameters.getInitialization().initialize(((RecurrentNeuralNetworkParameter) parameters).getHiddenLayer(i), ((RecurrentNeuralNetworkParameter) parameters).getHiddenLayer(i), random), new int[]{((RecurrentNeuralNetworkParameter) parameters).getHiddenLayer(i), ((RecurrentNeuralNetworkParameter) parameters).getHiddenLayer(i)}), false));
            }
            currentLength = ((RecurrentNeuralNetworkParameter) parameters).getHiddenLayer(i) + 1;
        }
        weights.add(new MultiplicationNode(true, false, new Tensor(parameters.getInitialization().initialize(currentLength, ((RecurrentNeuralNetworkParameter) parameters).getClassLabelSize(), random), new int[]{currentLength, ((RecurrentNeuralNetworkParameter) parameters).getClassLabelSize()}), false));
        ArrayList<ComputationalNode> currentOldLayers = new ArrayList<>();
        ArrayList<ComputationalNode> outputNodes = new ArrayList<>();
        for (int k = 0; k < timeStep; k++) {
            this.switches.add(new Switch());
            ArrayList<ComputationalNode> newOldLayers = new ArrayList<>();
            ComputationalNode input = new MultiplicationNode(false, true, false);
            inputNodes.add(input);
            ComputationalNode current = input;
            for (int i = 0; i < ((RecurrentNeuralNetworkParameter) parameters).size(); i++) {
                ComputationalNode aw;
                ComputationalNode aFunction;
                if (!currentOldLayers.isEmpty()) {
                    aw = this.addEdge(current, weights.get((i * 3)), false);
                    ComputationalNode oWithoutBias = this.addEdge(currentOldLayers.get(i), new RemoveBias(), false);
                    ComputationalNode ou = this.addEdge(oWithoutBias, recurrentWeights.get((i * 3)), false);
                    ComputationalNode awOu = this.addAdditionEdge(aw, ou, false);
                    ComputationalNode zt = this.addEdge(awOu, ((RecurrentNeuralNetworkParameter) parameters).getActivationFunction((i * 2)), false);
                    aw = this.addEdge(current, weights.get((i * 3) + 1), false);
                    ou = this.addEdge(oWithoutBias, recurrentWeights.get((i * 3) + 1), false);
                    awOu = this.addAdditionEdge(aw, ou, false);
                    ComputationalNode rt = this.addEdge(awOu, ((RecurrentNeuralNetworkParameter) parameters).getActivationFunction((i * 2) + 1), false);
                    aw = this.addEdge(current, weights.get((i * 3) + 2), false);
                    ComputationalNode rtHt1 = this.addEdge(rt, oWithoutBias, false, true);
                    ou = this.addEdge(rtHt1, recurrentWeights.get((i * 3) + 2), false);
                    awOu = this.addAdditionEdge(aw, ou, false);
                    ComputationalNode hTemp = this.addEdge(awOu, new Tanh(), false);
                    ComputationalNode minusZt = this.addEdge(zt, new Negation(), false);
                    ComputationalNode oneMinusZt = this.addEdge(minusZt, new AdditionByConstant(1.0), false);
                    aw = this.addEdge(oneMinusZt, oWithoutBias, false, true);
                    ou = this.addEdge(hTemp, zt, false, true);
                    aFunction = this.addAdditionEdge(aw, ou, true);
                } else {
                    aw = this.addEdge(current, weights.get((i * 3)), false);
                    ComputationalNode zt = this.addEdge(aw, ((RecurrentNeuralNetworkParameter) parameters).getActivationFunction((i * 2)), false);
                    aw = this.addEdge(current, weights.get((i * 3) + 2), false);
                    ComputationalNode hTemp = this.addEdge(aw, new Tanh(), false);
                    aFunction = this.addEdge(zt, hTemp, true, true);
                }
                current = aFunction;
                newOldLayers.add(aFunction);
            }
            currentOldLayers = newOldLayers;
            ComputationalNode node = this.addEdge(current, weights.get(weights.size() - 1), false);
            outputNodes.add(this.addEdge(node, switches.get(k), false));
        }
        ConcatenatedNode concatenatedNode = (ConcatenatedNode) this.concatEdges(outputNodes, 0);
        this.addEdge(concatenatedNode, new Softmax(), false);
        train(trainSet, parameters, random);
    }
}
