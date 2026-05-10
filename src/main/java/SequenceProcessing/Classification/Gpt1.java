package SequenceProcessing.Classification;

import Classification.Performance.ClassificationPerformance;
import ComputationalGraph.*;
import ComputationalGraph.Function.*;
import ComputationalGraph.Node.*;
import Dictionary.*;
import Math.Tensor;
import Math.Vector;
import SequenceProcessing.Functions.*;
import SequenceProcessing.Parameters.Gpt1Parameter;

import java.io.Serializable;
import java.util.*;

/**
 * Simplified GPT 1 style decoder only Transformer.
 *
 * The main difference from the original Transformer is that we remove the
 * encoder and the cross attention layer entirely. We only keep the masked
 * self attention so the model can only look at previous tokens (left to right).
 *
 * The data flow through the model looks like this:
 *   Input Embeddings + Positional Encoding
 *     -> Masked Multi Head Self Attention
 *     -> Add residual
 *     -> Layer Norm
 *     -> Feed Forward Network
 *     -> Add residual
 *     -> Layer Norm
 *     -> Linear projection to vocab size
 *     -> Softmax
 *
 * We normalize after adding the residual, i.e. LayerNorm(x + Sublayer(x)).
 *
 * Note: this is a simplified version for educational purposes,
 * not a full reproduction of the original GPT 1 paper.
 */
public class Gpt1 extends ComputationalGraph implements Serializable {

    private final VectorizedDictionary dictionary;
    private int startIndex;
    private int endIndex;

    /**
     * Constructor. Takes model parameters and a dictionary of word embeddings.
     * Also finds the start and end token indices in the dictionary so we can
     * use them later during autoregressive generation.
     */
    public Gpt1(NeuralNetworkParameter parameter, VectorizedDictionary dictionary) {
        super(parameter);
        this.dictionary = dictionary;
        for (int k = 0; k < this.dictionary.size(); k++) {
            if (this.dictionary.getWord(k).getName().equals("<S>")) {
                this.startIndex = k;
            } else if (this.dictionary.getWord(k).getName().equals("</S>")) {
                this.endIndex = k;
            }
        }
    }

    /**
     * Adds sinusoidal positional encoding to the embeddings.
     * Since transformers don't have any notion of token order
     * (unlike RNNs), we add sine/cosine signals so the model knows
     * which position each token is at.
     *
     * Even dimensions use sin, odd dimensions use cos.
     */
    private Tensor positionalEncoding(Tensor tensor, int wordEmbeddingLength) {
        ArrayList<Double> values = new ArrayList<>();
        for (int i = 0; i < tensor.getShape()[0]; i++) {
            for (int j = 0; j < tensor.getShape()[1]; j++) {
                double val = tensor.getValue(new int[]{i, j});
                if (j % 2 == 0) {
                    values.add(val + Math.sin((i + 1.0) / Math.pow(10000, (j + 0.0) / wordEmbeddingLength)));
                } else {
                    values.add(val + Math.cos((i + 1.0) / Math.pow(10000, (j - 1.0) / wordEmbeddingLength)));
                }
            }
        }
        return new Tensor(values, tensor.getShape());
    }

    /**
     * Reads a flat input tensor and splits it into embedding values and class labels.
     * The format is [emb_1, emb_2, ..., emb_d, label] repeated for each token.
     * After extracting the embeddings, we reshape them into a 2D tensor and
     * add positional encoding on top.
     *
     * Unlike the full Transformer which needs separate encoder and decoder inputs,
     * here we only have one input since there is no encoder.
     */
    private ArrayList<Integer> createInputTensors(Tensor instance, ComputationalNode input, int wordEmbeddingLength) {
        ArrayList<Integer> classLabels = new ArrayList<>();
        ArrayList<Double> values = new ArrayList<>();
        int curLength = 0;
        for (int i = 0; i < instance.getShape()[0]; i++) {
            if ((curLength + 1) % (wordEmbeddingLength + 1) == 0) {
                // every (d+1)-th value is the class label for that token
                classLabels.add((int) instance.getValue(new int[]{i}));
            } else {
                values.add(instance.getValue(new int[]{i}));
            }
            curLength++;
        }
        input.setValue(new Tensor(values, new int[]{values.size() / wordEmbeddingLength, wordEmbeddingLength}));
        input.setValue(positionalEncoding(input.getValue(), wordEmbeddingLength));
        return classLabels;
    }

    /**
     * Layer normalization. We use this after each residual connection.
     * The formula is: LN(x) = gamma * (x - mean) / sqrt(variance + epsilon) + beta
     *
     * gamma and beta are learnable parameters that let the model scale and shift
     * the normalized values. epsilon is a small number to avoid dividing by zero.
     *
     * lnSize keeps track of which gamma/beta values we've used so far,
     * since each layer norm block has its own set of parameters.
     */
    private ComputationalNode layerNormalization(ComputationalNode input, Gpt1Parameter parameter, int[] lnSize) {
        ArrayList<Double> data = new ArrayList<>();
        // compute mean
        ComputationalNode inputMean = this.addEdge(input, new Mean());
        ComputationalNode meanNeg = this.addEdge(inputMean, new Negation());
        // subtract mean from input
        ComputationalNode inputMinusMean = this.addAdditionEdge(input, meanNeg, false);
        // compute variance, then 1/sqrt(var + eps)
        ComputationalNode variance = this.addEdge(inputMinusMean, new Variance());
        ComputationalNode rootVariance = this.addEdge(variance, new SquareRoot(parameter.getEpsilon()));
        ComputationalNode inverseRootVariance = this.addEdge(rootVariance, new Inverse());
        // normalize: (x - mean) / sqrt(var + eps)
        ComputationalNode lnValue = this.addEdge(inputMinusMean, inverseRootVariance, false, true);
        // scale by gamma
        for (int j = 0; j < parameter.getL(); j++) {
            data.add(parameter.getGammaValue(lnSize[0]));
        }
        lnSize[0]++;
        ComputationalNode gamma = new MultiplicationNode(true, false, new Tensor(data, new int[]{1, parameter.getL()}), true);
        ComputationalNode lnGamma = this.addEdge(lnValue, gamma);
        // shift by beta
        data.clear();
        for (int j = 0; j < parameter.getL(); j++) {
            data.add(parameter.getBetaValue(lnSize[1]));
        }
        lnSize[1]++;
        ComputationalNode beta = new ComputationalNode(true, false, new Tensor(data, new int[]{1, parameter.getL()}));
        return this.addAdditionEdge(lnGamma, beta, false);
    }

    /**
     * Masked multi head self attention. This is the core of the GPT decoder.
     *
     * For each attention head we create separate Q, K, V projections from the input,
     * then compute: Attention = softmax(mask(Q * K^T / sqrt(dk))) * V
     *
     * The mask makes sure that position i can only attend to positions <= i,
     * which is what makes this autoregressive (the model can't cheat by
     * looking at future tokens).
     *
     * We run N heads in parallel (each looking at dk dimensions) and
     * concatenate the results at the end.
     */
    private ArrayList<ComputationalNode> maskedMultiHeadAttention(ComputationalNode input, Gpt1Parameter parameter, Random random) {
        ArrayList<ComputationalNode> nodes = new ArrayList<>();
        for (int i = 0; i < parameter.getN(); i++) {
            // project input into key, query, and value for this head
            ComputationalNode wk = new MultiplicationNode(new Tensor(parameter.initializeWeights(parameter.getL(), parameter.getDk(), random), new int[]{parameter.getL(), parameter.getDk()}));
            ComputationalNode k = this.addEdge(input, wk);
            ComputationalNode wq = new MultiplicationNode(new Tensor(parameter.initializeWeights(parameter.getL(), parameter.getDk(), random), new int[]{parameter.getL(), parameter.getDk()}));
            ComputationalNode q = this.addEdge(input, wq);
            ComputationalNode wv = new MultiplicationNode(new Tensor(parameter.initializeWeights(parameter.getL(), parameter.getDk(), random), new int[]{parameter.getL(), parameter.getDk()}));
            ComputationalNode v = this.addEdge(input, wv);
            // Q * K^T
            ComputationalNode kTranspose = this.addEdge(k, new Transpose());
            ComputationalNode qk = this.addEdge(q, kTranspose, false, false);
            // scale by 1/sqrt(dk) to keep gradients stable
            ComputationalNode qkDk = this.addEdge(qk, new MultiplyByConstant(1.0 / Math.sqrt(parameter.getDk())));
            // apply causal mask (set future positions to -inf so softmax gives them 0)
            ComputationalNode mQkDk = this.addEdge(qkDk, new Mask());
            ComputationalNode sQkDk = this.addEdge(mQkDk, new Softmax());
            // multiply attention weights by values
            ComputationalNode attention = this.addEdge(sQkDk, v);
            nodes.add(attention);
        }
        return nodes;
    }

    /**
     * Feed forward network. This is applied to each position
     * independently (same weights, but each token goes through it separately).
     * Basically two linear layers with an activation function in between.
     */
    private ComputationalNode feedforwardNeuralNetwork(ComputationalNode current, int currentLayerSize, Gpt1Parameter parameter, Random random) {
        for (int i = 0; i < parameter.getHiddenLayerSize(); i++) {
            ComputationalNode hiddenWeight = new MultiplicationNode(new Tensor(parameter.initializeWeights(currentLayerSize, parameter.getHiddenLayer(i), random), new int[]{currentLayerSize, parameter.getHiddenLayer(i)}));
            ComputationalNode hiddenLayer = this.addEdge(current, hiddenWeight);
            current = this.addEdge(hiddenLayer, parameter.getActivationFunction(i), true);
            currentLayerSize = parameter.getHiddenLayer(i) + 1;
        }
        ComputationalNode outputWeight = new MultiplicationNode(new Tensor(parameter.initializeWeights(currentLayerSize, parameter.getL(), random), new int[]{currentLayerSize, parameter.getL()}));
        return this.addEdge(current, outputWeight);
    }

    /**
     * Builds the computational graph and trains the model.
     *
     * The graph follows this architecture:
     *   1. Input embeddings + positional encoding
     *   2. Masked multi head self attention
     *   3. Residual add + layer norm
     *   4. Feed forward network
     *   5. Residual add + layer norm
     *   6. Linear projection to vocab size + softmax
     *
     * After building the graph, we train for the specified number of epochs.
     * Each epoch shuffles the data, then does forward pass + backprop for
     * each training instance.
     */
    @Override
    public void train(ArrayList<Tensor> trainSet) {
        Gpt1Parameter parameter = (Gpt1Parameter) this.parameters;
        int[] lnSize = new int[2];
        Random random = new Random(parameter.getSeed());

        // --- Build the computational graph ---

        // Input node (will hold embeddings + positional encoding)
        ComputationalNode input = new MultiplicationNode(false, true);
        this.inputNodes.add(input);

        // Masked multi head self attention
        // Each head produces its own attention output, then we concat them all
        ConcatenatedNode concatenatedNode = (ConcatenatedNode) this.concatEdges(maskedMultiHeadAttention(input, parameter, random), 1);
        // Project concatenated heads back to embedding dimension
        ComputationalNode wo = new MultiplicationNode(new Tensor(parameter.initializeWeights(parameter.getL(), parameter.getL(), random), new int[]{parameter.getL(), parameter.getL()}));
        ComputationalNode attnOutput = this.addEdge(concatenatedNode, wo);

        // Residual connection + layer norm (normalize after adding)
        ComputationalNode afterAttnResidual = this.addAdditionEdge(input, attnOutput, false);
        ComputationalNode afterAttn = layerNormalization(afterAttnResidual, parameter, lnSize);

        // Feed forward network
        ComputationalNode ffOutput = feedforwardNeuralNetwork(afterAttn, parameter.getL(), parameter, random);

        // Residual connection + layer norm again
        ComputationalNode afterFFResidual = this.addAdditionEdge(afterAttn, ffOutput, false);
        ComputationalNode afterFF = layerNormalization(afterFFResidual, parameter, lnSize);

        // Final linear projection to vocabulary size, then softmax to get probabilities
        ComputationalNode wLinear = new MultiplicationNode(new Tensor(parameter.initializeWeights(parameter.getL(), parameter.getV(), random), new int[]{parameter.getL(), parameter.getV()}));
        ComputationalNode linear = this.addEdge(afterFF, wLinear);
        this.outputNode = this.addEdge(linear, new Softmax());

        // Set up loss computation
        ComputationalNode classLabelNode = new ComputationalNode();
        this.addLoss(classLabelNode);

        // --- Training loop ---
        for (int i = 0; i < parameter.getEpoch(); i++) {
            this.shuffle(trainSet, random);
            for (Tensor instance : trainSet) {
                // Parse input and get the gold labels
                ArrayList<Integer> classLabels = createInputTensors(instance, this.inputNodes.get(0), parameter.getL() - 1);
                // Convert labels to one hot vectors for cross entropy loss
                ArrayList<Double> classLabelValues = new ArrayList<>();
                for (Integer classLabel : classLabels) {
                    for (int j = 0; j < parameter.getV(); j++) {
                        if (j == classLabel) {
                            classLabelValues.add(1.0);
                        } else {
                            classLabelValues.add(0.0);
                        }
                    }
                }
                classLabelNode.setValue(new Tensor(classLabelValues, new int[]{classLabels.size(), parameter.getV()}));
                this.forwardCalculation();
                this.backpropagation();
            }
            parameter.getOptimizer().setLearningRate();
        }
    }

    /**
     * Helper for autoregressive generation. Appends a new token's embedding
     * to the growing input sequence and adds positional encoding for it.
     */
    private void setInputNode(int bound, Vector vector, ComputationalNode node) {
        ArrayList<Double> data = new ArrayList<>();
        if (node.getValue() != null) {
            data = (ArrayList<Double>) node.getValue().getData();
        }
        for (int i = 0; i < vector.size(); i++) {
            if (i % 2 == 0) {
                data.add(vector.getValue(i) + Math.sin((bound + 0.0) / Math.pow(10000, (i + 0.0) / vector.size())));
            } else {
                data.add(vector.getValue(i) + Math.cos((bound + 0.0) / Math.pow(10000, (i - 1.0) / vector.size())));
            }
        }
        node.setValue(new Tensor(data, new int[]{bound, vector.size()}));
    }

    /**
     * Tests the model by doing autoregressive generation on the test set.
     * We start with the start token, predict the next token, feed it back in,
     * and keep going until we hit the end token. Then we compare our
     * predictions against the actual labels and compute accuracy.
     */
    @Override
    public ClassificationPerformance test(ArrayList<Tensor> testSet) {
        int count = 0, total = 0;
        for (Tensor instance : testSet) {
            ArrayList<Double> classLabels;
            ArrayList<Integer> goldClassLabels = createInputTensors(instance, this.inputNodes.get(0), ((VectorizedWord) this.dictionary.getWord(0)).getVector().size());
            int j = 1;
            int currentWordIndex = this.startIndex;
            this.inputNodes.get(0).setValue(null);
            do {
                setInputNode(j, ((VectorizedWord) this.dictionary.getWord(currentWordIndex)).getVector(), this.inputNodes.get(0));
                classLabels = this.predict();
                if (goldClassLabels.size() >= classLabels.size() && classLabels.get(classLabels.size() - 1).intValue() == goldClassLabels.get(classLabels.size() - 1)) {
                    count++;
                }
                total++;
                j++;
                currentWordIndex = classLabels.get(classLabels.size() - 1).intValue();
            } while (currentWordIndex != this.endIndex);
            if (classLabels.size() < goldClassLabels.size()) {
                total += goldClassLabels.size() - classLabels.size();
            }
        }
        return new ClassificationPerformance((count + 0.00) / total);
    }

    /**
     * Gets the predicted class for each position by finding the index with
     * the highest probability (argmax) in each row of the output.
     */
    @Override
    protected ArrayList<Double> getOutputValue() {
        ArrayList<Double> classLabels = new ArrayList<>();
        Tensor value = outputNode.getValue();
        for (int i = 0; i < value.getShape()[0]; i++) {
            double max = Double.MIN_VALUE;
            double index = -1;
            for (int j = 0; j < value.getShape()[1]; j++) {
                if (value.getValue(new int[]{i, j}) > max) {
                    max = value.getValue(new int[]{i, j});
                    index = j;
                }
            }
            classLabels.add(index);
        }
        return classLabels;
    }
}
