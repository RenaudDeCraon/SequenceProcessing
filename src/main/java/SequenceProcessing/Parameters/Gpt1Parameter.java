package SequenceProcessing.Parameters;

import ComputationalGraph.Initialization.Initialization;
import ComputationalGraph.NeuralNetworkParameter;

import java.io.Serializable;
import java.util.ArrayList;

public class Gpt1Parameter extends NeuralNetworkParameter implements Serializable {

    private final int L;
    private final int N;
    private final int V;
    private final double epsilon;
    private final ArrayList<Integer> hiddenLayers;
    private final ArrayList<Object> activationFunctions;
    private final ArrayList<Double> gammaValues;
    private final ArrayList<Double> betaValues;

    public Gpt1Parameter(int seed, int epoch, ComputationalGraph.Optimizer.Optimizer optimizer,
                         Initialization initialization, ComputationalGraph.Loss.Loss loss,
                         int wordEmbeddingLength, int headCount, int vocabSize,
                         double epsilon, ArrayList<Integer> hiddenLayers,
                         ArrayList<Object> activationFunctions,
                         ArrayList<Double> gammaValues, ArrayList<Double> betaValues) {
        super(seed, epoch, optimizer, initialization, loss, 0.0, -1);
        this.L = wordEmbeddingLength + 1;
        this.N = headCount;
        this.V = vocabSize;
        this.epsilon = epsilon;
        this.hiddenLayers = hiddenLayers;
        this.activationFunctions = activationFunctions;
        this.gammaValues = gammaValues;
        this.betaValues = betaValues;
    }

    public int getL() {
        return L;
    }

    public int getN() {
        return N;
    }

    public int getV() {
        return V;
    }

    public int getDk() {
        return L / N;
    }

    public double getEpsilon() {
        return epsilon;
    }

    public double getGammaValue(int index) {
        return gammaValues.get(index);
    }

    public double getBetaValue(int index) {
        return betaValues.get(index);
    }

    public int getHiddenLayer(int index) {
        return hiddenLayers.get(index);
    }

    public int getHiddenLayerSize() {
        return hiddenLayers.size();
    }

    public Object getActivationFunction(int index) {
        return activationFunctions.get(index);
    }
}
