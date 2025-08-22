package SequenceProcessing.Classification;

import Classification.Parameter.Parameter;
import Classification.Performance.ClassificationPerformance;
import ComputationalGraph.*;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;

import Math.*;
import SequenceProcessing.Functions.RemoveBias;
import SequenceProcessing.Functions.Switch;
import SequenceProcessing.Parameters.*;

public class RecurrentNeuralNetworkModel extends ComputationalGraph implements Serializable {

    private final int wordEmbeddingLength;
    private final ArrayList<Switch> switches;

    public RecurrentNeuralNetworkModel(int wordEmbeddingLength) {
        this.wordEmbeddingLength = wordEmbeddingLength;
        this.switches = new ArrayList<>();
    }

    private void createInputTensors(Tensor instance, ArrayList<Integer> classLabels) {
        int timeStep = (instance.getShape()[0] / (wordEmbeddingLength + 1));
        int j = 0;
        for (int i = 0; i < this.inputNodes.size(); i++) {
            if (i < timeStep) {
                this.switches.get(i).setTurn(true);
                ArrayList<Double> values = new ArrayList<>();
                for (int k = 0; k < wordEmbeddingLength; k++) {
                    values.add(instance.getValue(new int[]{j}));
                    j++;
                }
                classLabels.add((int) instance.getValue(new int[]{j}));
                j++;
                inputNodes.get(i).setValue(new Tensor(values, new int[]{1, values.size()}));
            } else {
                this.switches.get(i).setTurn(false);
                ArrayList<Double> values = new ArrayList<>();
                for (int k = 0; k < wordEmbeddingLength; k++) {
                    values.add(0.0);
                    j++;
                }
                classLabels.add(0);
                j++;
                inputNodes.get(i).setValue(new Tensor(values, new int[]{1, values.size()}));
            }
        }
    }

    // Many-to-Many RNN
    @Override
    public void train(ArrayList<Tensor> trainSet, Parameter parameters) {
        Random random = new Random(1);
        int timeStep = -1;
        for (Tensor tensor : trainSet) {
            int size = tensor.getShape()[0];
            if (timeStep < size / (wordEmbeddingLength + 1)) {
                timeStep = size / (wordEmbeddingLength + 1);
            }
        }
        ArrayList<ComputationalNode> weights = new ArrayList<>();
        ArrayList<ComputationalNode> recurrentWeights = new ArrayList<>();
        ArrayList<Double> data = new ArrayList<>();
        int currentLength = wordEmbeddingLength + 1;
        for (int i = 0; i < ((RecurrentNeuralNetworkParameter) parameters).size(); i++) {
            data.clear();
            for (int j = 0; j < currentLength * ((RecurrentNeuralNetworkParameter) parameters).getHiddenLayer(i); j++) {
                data.add(-0.01 + (0.02 * random.nextDouble()));
            }
            weights.add(new MultiplicationNode(true, false, new Tensor(data, new int[]{currentLength, ((RecurrentNeuralNetworkParameter) parameters).getHiddenLayer(i)}), false));
            data.clear();
            for (int j = 0; j < ((RecurrentNeuralNetworkParameter) parameters).getHiddenLayer(i) * ((RecurrentNeuralNetworkParameter) parameters).getHiddenLayer(i); j++) {
                data.add(-0.01 + (0.02 * random.nextDouble()));
            }
            recurrentWeights.add(new MultiplicationNode(true, false, new Tensor(data, new int[]{((RecurrentNeuralNetworkParameter) parameters).getHiddenLayer(i), ((RecurrentNeuralNetworkParameter) parameters).getHiddenLayer(i)}), false));
            currentLength = ((RecurrentNeuralNetworkParameter) parameters).getHiddenLayer(i) + 1;
        }
        data.clear();
        for (int j = 0; j < currentLength * ((RecurrentNeuralNetworkParameter) parameters).getClassLabelSize(); j++) {
            data.add(-0.01 + (0.02 * random.nextDouble()));
        }
        weights.add(new MultiplicationNode(true, false, new Tensor(data, new int[]{currentLength, ((RecurrentNeuralNetworkParameter) parameters).getClassLabelSize()}), false));
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
                    aw = this.addEdge(current, weights.get(i), false);
                    ComputationalNode oWithoutBias = this.addEdge(currentOldLayers.get(i), new RemoveBias(), false);
                    ComputationalNode ou = this.addEdge(oWithoutBias, recurrentWeights.get(i), false);
                    ComputationalNode a = this.addAdditionEdge(aw, ou, false);
                    aFunction = this.addEdge(a, ((RecurrentNeuralNetworkParameter) parameters).getActivationFunction(i), true);
                } else {
                    aw = this.addEdge(current, weights.get(i), false);
                    aFunction = this.addEdge(aw, ((RecurrentNeuralNetworkParameter) parameters).getActivationFunction(i), true);
                }
                current = this.addEdge(aFunction, switches.get(k), false);
                newOldLayers.add(current);
            }
            currentOldLayers = (ArrayList<ComputationalNode>) newOldLayers.clone();
            ComputationalNode output = this.addEdge(current, weights.get(weights.size() - 1), false);
            outputNodes.add(this.addEdge(output, new Softmax(), false));
        }
        this.concatEdges(outputNodes, 0);
        // Training
        for (int i = 0; i < ((RecurrentNeuralNetworkParameter) parameters).getEpoch(); i++) {
            System.out.println("Epoch: " + (i + 1));
            // Shuffle
            for (int j = 0; j < trainSet.size(); j++) {
                int i1 = random.nextInt(trainSet.size());
                int i2 = random.nextInt(trainSet.size());
                Tensor tmp = trainSet.get(i1);
                trainSet.set(i1, trainSet.get(i2));
                trainSet.set(i2, tmp);
            }
            ArrayList<Integer> classLabels = new ArrayList<>();
            for (Tensor instance : trainSet) {
                createInputTensors(instance, classLabels);
                this.forwardCalculation();
                this.backpropagation(((RecurrentNeuralNetworkParameter) parameters).getLearningRate(), classLabels);
                classLabels.clear();
            }
            ((RecurrentNeuralNetworkParameter) parameters).setLearningRate();
        }
    }

    @Override
    protected ArrayList<Integer> getClassLabels(ComputationalNode outputNode) {
        ArrayList<Integer> classLabels = new ArrayList<>();
        for (int i = 0; i < outputNode.getValue().getShape()[0]; i++) {
            int index = -1;
            double max = Double.MIN_VALUE;
            for (int j = 0; j < outputNode.getValue().getShape()[1]; j++) {
                if (max < outputNode.getValue().getValue(new int[]{i, j})) {
                    max = outputNode.getValue().getValue(new int[]{i, j});
                    index = j;
                }
            }
            classLabels.add(index);
        }
        return classLabels;
    }

    @Override
    public ClassificationPerformance test(ArrayList<Tensor> testSet) {
        int count = 0, total = 0;
        for (Tensor instance : testSet) {
            ArrayList<Integer> goldClassLabels = new ArrayList<>();
            createInputTensors(instance, goldClassLabels);
            ArrayList<Integer> classLabels = this.predict();
            for (int j = 0; j < (instance.getShape()[0] / (wordEmbeddingLength + 1)); j++) {
                if (goldClassLabels.get(j).equals(classLabels.get(j))) {
                    count++;
                }
                total++;
            }
        }
        return new ClassificationPerformance((count + 0.0) / total);
    }
}
