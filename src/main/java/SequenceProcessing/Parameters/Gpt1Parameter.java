package SequenceProcessing.Parameters;

import ComputationalGraph.Initialization.Initialization;
import ComputationalGraph.NeuralNetworkParameter;

import java.io.Serializable;
import java.util.ArrayList;

/**
 * Holds the hyperparameters for the GPT 1 style decoder only Transformer.
 *
 * Compared to TransformerParameter (which has separate encoder/decoder params),
 * this is simpler because GPT 1 only has a decoder with masked self attention.
 * So we just need one set of hidden layers, activation functions, and
 * layer norm parameters instead of two.
 */
public class Gpt1Parameter extends NeuralNetworkParameter implements Serializable {

    private final int L;  // embedding dimension (wordEmbeddingLength + 1 for bias)
    private final int N;  // number of attention heads
    private final int V;  // vocabulary size
    private final double epsilon;  // small constant for layer norm stability
    private final ArrayList<Integer> hiddenLayers;
    private final ArrayList<Object> activationFunctions;
    private final ArrayList<Double> gammaValues;  // scale params for layer norm
    private final ArrayList<Double> betaValues;   // shift params for layer norm

    /**
     * Creates a new parameter set for the GPT 1 model.
     *
     * @param seed                  random seed for reproducibility
     * @param epoch                 how many times to go through the training data
     * @param optimizer             optimizer to use (e.g. AdamW)
     * @param initialization        how to initialize weights (e.g. random)
     * @param loss                  loss function (e.g. cross entropy)
     * @param wordEmbeddingLength   size of word embeddings
     * @param multiHeadAttentionLength  number of attention heads
     * @param vocabularyLength      number of words in the vocabulary
     * @param epsilon               epsilon for layer norm (usually something like 1e-9)
     * @param hiddenLayers          sizes of hidden layers in the feed forward network
     * @param activationFunctions   activation functions for each hidden layer
     * @param gammaValues           gamma (scale) for each layer norm
     * @param betaValues            beta (shift) for each layer norm
     */
    public Gpt1Parameter(int seed, int epoch, ComputationalGraph.Optimizer.Optimizer optimizer,
                         Initialization initialization, ComputationalGraph.Loss.Loss loss,
                         int wordEmbeddingLength, int multiHeadAttentionLength, int vocabularyLength,
                         double epsilon, ArrayList<Integer> hiddenLayers,
                         ArrayList<Object> activationFunctions,
                         ArrayList<Double> gammaValues, ArrayList<Double> betaValues) {
        super(seed, epoch, optimizer, initialization, loss, 0.0, -1);
        this.L = wordEmbeddingLength + 1;
        this.N = multiHeadAttentionLength;
        this.V = vocabularyLength;
        this.epsilon = epsilon;
        this.hiddenLayers = hiddenLayers;
        this.activationFunctions = activationFunctions;
        this.gammaValues = gammaValues;
        this.betaValues = betaValues;
    }

    /**
     * Returns gamma value for the layer norm at the given index.
     */
    public double getGammaValue(int index) {
        return gammaValues.get(index);
    }

    /**
     * Returns beta value for the layer norm at the given index.
     */
    public double getBetaValue(int index) {
        return betaValues.get(index);
    }

    /**
     * Returns epsilon used in layer norm to avoid division by zero.
     */
    public double getEpsilon() {
        return epsilon;
    }

    /**
     * Returns the dimension of each attention head's key/query vectors.
     * This is just L / N (total dim divided by number of heads).
     */
    public int getDk() {
        return L / N;
    }

    /**
     * Returns the full embedding dimension L.
     */
    public int getL() {
        return L;
    }

    /**
     * Returns the number of attention heads.
     */
    public int getN() {
        return N;
    }

    /**
     * Returns the vocabulary size.
     */
    public int getV() {
        return V;
    }

    /**
     * Returns the size of a specific hidden layer in the feed forward network.
     */
    public int getHiddenLayer(int index) {
        return hiddenLayers.get(index);
    }

    /**
     * Returns the activation function for a specific layer.
     */
    public Object getActivationFunction(int index) {
        return activationFunctions.get(index);
    }

    /**
     * Returns how many hidden layers the feed forward network has.
     */
    public int getHiddenLayerSize() {
        return hiddenLayers.size();
    }
}
