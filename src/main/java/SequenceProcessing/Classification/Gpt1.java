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

public class Gpt1 extends ComputationalGraph implements Serializable {

    private final VectorizedDictionary dictionary;
    private int startIndex;
    private int endIndex;

    public Gpt1(NeuralNetworkParameter parameter, VectorizedDictionary dictionary) {
        super(parameter);
        this.dictionary = dictionary;
        for (int i = 0; i < this.dictionary.size(); i++) {
            String name = this.dictionary.getWord(i).getName();
            if (name.equals("<S>")) {
                this.startIndex = i;
            } else if (name.equals("</S>")) {
                this.endIndex = i;
            }
        }
    }

    private Tensor positionalEncoding(Tensor tensor, int embeddingSize) {
        ArrayList<Double> result = new ArrayList<>();
        int rows = tensor.getShape()[0];
        int cols = tensor.getShape()[1];
        for (int pos = 0; pos < rows; pos++) {
            for (int dim = 0; dim < cols; dim++) {
                double original = tensor.getValue(new int[]{pos, dim});
                double angle = (pos + 1.0) / Math.pow(10000, (dim % 2 == 0 ? dim : dim - 1.0) / embeddingSize);
                if (dim % 2 == 0) {
                    result.add(original + Math.sin(angle));
                } else {
                    result.add(original + Math.cos(angle));
                }
            }
        }
        return new Tensor(result, tensor.getShape());
    }

    private ArrayList<Integer> createInputTensors(Tensor instance, ComputationalNode inputNode, int embeddingSize) {
        ArrayList<Integer> labels = new ArrayList<>();
        ArrayList<Double> embeddings = new ArrayList<>();
        int counter = 0;
        for (int i = 0; i < instance.getShape()[0]; i++) {
            double val = instance.getValue(new int[]{i});
            if ((counter + 1) % (embeddingSize + 1) == 0) {
                labels.add((int) val);
            } else {
                embeddings.add(val);
            }
            counter++;
        }
        int tokenCount = embeddings.size() / embeddingSize;
        inputNode.setValue(new Tensor(embeddings, new int[]{tokenCount, embeddingSize}));
        inputNode.setValue(positionalEncoding(inputNode.getValue(), embeddingSize));
        return labels;
    }

    private ComputationalNode layerNormalization(ComputationalNode input, Gpt1Parameter parameter, int[] lnIdx) {
        ComputationalNode mean = this.addEdge(input, new Mean());
        ComputationalNode negMean = this.addEdge(mean, new Negation());
        ComputationalNode centered = this.addAdditionEdge(input, negMean, false);
        ComputationalNode var = this.addEdge(centered, new Variance());
        ComputationalNode sqrtVar = this.addEdge(var, new SquareRoot(parameter.getEpsilon()));
        ComputationalNode invSqrtVar = this.addEdge(sqrtVar, new Inverse());
        ComputationalNode normalized = this.addEdge(centered, invSqrtVar, false, true);
        ArrayList<Double> gammaData = new ArrayList<>();
        for (int i = 0; i < parameter.getL(); i++) {
            gammaData.add(parameter.getGammaValue(lnIdx[0]));
        }
        lnIdx[0]++;
        ComputationalNode gammaNode = new MultiplicationNode(true, false, new Tensor(gammaData, new int[]{1, parameter.getL()}), true);
        ComputationalNode scaled = this.addEdge(normalized, gammaNode);
        ArrayList<Double> betaData = new ArrayList<>();
        for (int i = 0; i < parameter.getL(); i++) {
            betaData.add(parameter.getBetaValue(lnIdx[1]));
        }
        lnIdx[1]++;
        ComputationalNode betaNode = new ComputationalNode(true, false, new Tensor(betaData, new int[]{1, parameter.getL()}));
        return this.addAdditionEdge(scaled, betaNode, false);
    }

    private ArrayList<ComputationalNode> maskedMultiHeadAttention(ComputationalNode input, Gpt1Parameter parameter, Random random) {
        ArrayList<ComputationalNode> heads = new ArrayList<>();
        int L = parameter.getL();
        int dk = parameter.getDk();
        for (int h = 0; h < parameter.getN(); h++) {
            ComputationalNode wk = new MultiplicationNode(new Tensor(parameter.initializeWeights(L, dk, random), new int[]{L, dk}));
            ComputationalNode wq = new MultiplicationNode(new Tensor(parameter.initializeWeights(L, dk, random), new int[]{L, dk}));
            ComputationalNode wv = new MultiplicationNode(new Tensor(parameter.initializeWeights(L, dk, random), new int[]{L, dk}));
            ComputationalNode k = this.addEdge(input, wk);
            ComputationalNode q = this.addEdge(input, wq);
            ComputationalNode v = this.addEdge(input, wv);
            ComputationalNode kt = this.addEdge(k, new Transpose());
            ComputationalNode scores = this.addEdge(q, kt, false, false);
            ComputationalNode scaledScores = this.addEdge(scores, new MultiplyByConstant(1.0 / Math.sqrt(dk)));
            ComputationalNode masked = this.addEdge(scaledScores, new Mask());
            ComputationalNode weights = this.addEdge(masked, new Softmax());
            heads.add(this.addEdge(weights, v));
        }
        return heads;
    }

    private ComputationalNode feedforwardNeuralNetwork(ComputationalNode input, int inputSize, Gpt1Parameter parameter, Random random) {
        ComputationalNode current = input;
        int size = inputSize;
        for (int i = 0; i < parameter.getHiddenLayerSize(); i++) {
            int hiddenSize = parameter.getHiddenLayer(i);
            ComputationalNode w = new MultiplicationNode(new Tensor(parameter.initializeWeights(size, hiddenSize, random), new int[]{size, hiddenSize}));
            ComputationalNode layer = this.addEdge(current, w);
            current = this.addEdge(layer, parameter.getActivationFunction(i), true);
            size = hiddenSize + 1;
        }
        ComputationalNode wOut = new MultiplicationNode(new Tensor(parameter.initializeWeights(size, parameter.getL(), random), new int[]{size, parameter.getL()}));
        return this.addEdge(current, wOut);
    }

    @Override
    public void train(ArrayList<Tensor> trainSet) {
        Gpt1Parameter parameter = (Gpt1Parameter) this.parameters;
        Random random = new Random(parameter.getSeed());
        int[] lnIdx = new int[2];
        int L = parameter.getL();
        int V = parameter.getV();
        // Decoder Block
        ComputationalNode input = new MultiplicationNode(false, true);
        this.inputNodes.add(input);
        ArrayList<ComputationalNode> attnHeads = maskedMultiHeadAttention(input, parameter, random);
        ConcatenatedNode concat = (ConcatenatedNode) this.concatEdges(attnHeads, 1);
        ComputationalNode wo = new MultiplicationNode(new Tensor(parameter.initializeWeights(L, L, random), new int[]{L, L}));
        ComputationalNode attnOut = this.addEdge(concat, wo);
        ComputationalNode residual1 = this.addAdditionEdge(input, attnOut, false);
        ComputationalNode norm1 = layerNormalization(residual1, parameter, lnIdx);
        ComputationalNode ffOut = feedforwardNeuralNetwork(norm1, L, parameter, random);
        ComputationalNode residual2 = this.addAdditionEdge(norm1, ffOut, false);
        ComputationalNode norm2 = layerNormalization(residual2, parameter, lnIdx);
        ComputationalNode wFinal = new MultiplicationNode(new Tensor(parameter.initializeWeights(L, V, random), new int[]{L, V}));
        ComputationalNode logits = this.addEdge(norm2, wFinal);
        this.outputNode = this.addEdge(logits, new Softmax());
        ComputationalNode labelNode = new ComputationalNode();
        this.addLoss(labelNode);
        // Training
        for (int epoch = 0; epoch < parameter.getEpoch(); epoch++) {
            this.shuffle(trainSet, random);
            for (Tensor instance : trainSet) {
                ArrayList<Integer> labels = createInputTensors(instance, this.inputNodes.get(0), L - 1);
                ArrayList<Double> oneHot = new ArrayList<>();
                for (int label : labels) {
                    for (int c = 0; c < V; c++) {
                        oneHot.add(c == label ? 1.0 : 0.0);
                    }
                }
                labelNode.setValue(new Tensor(oneHot, new int[]{labels.size(), V}));
                this.forwardCalculation();
                this.backpropagation();
            }
            parameter.getOptimizer().setLearningRate();
        }
    }

    private void setInputNode(int step, Vector embedding, ComputationalNode node) {
        ArrayList<Double> data = new ArrayList<>();
        if (node.getValue() != null) {
            data = (ArrayList<Double>) node.getValue().getData();
        }
        int vecSize = embedding.size();
        for (int i = 0; i < vecSize; i++) {
            double posEnc;
            if (i % 2 == 0) {
                posEnc = Math.sin((step + 0.0) / Math.pow(10000, (i + 0.0) / vecSize));
            } else {
                posEnc = Math.cos((step + 0.0) / Math.pow(10000, (i - 1.0) / vecSize));
            }
            data.add(embedding.getValue(i) + posEnc);
        }
        node.setValue(new Tensor(data, new int[]{step, vecSize}));
    }

    @Override
    public ClassificationPerformance test(ArrayList<Tensor> testSet) {
        int correct = 0;
        int total = 0;
        int vecSize = ((VectorizedWord) this.dictionary.getWord(0)).getVector().size();
        for (Tensor instance : testSet) {
            ArrayList<Integer> goldLabels = createInputTensors(instance, this.inputNodes.get(0), vecSize);
            this.inputNodes.get(0).setValue(null);
            int step = 1;
            int wordIdx = this.startIndex;
            ArrayList<Double> predicted;
            do {
                Vector vec = ((VectorizedWord) this.dictionary.getWord(wordIdx)).getVector();
                setInputNode(step, vec, this.inputNodes.get(0));
                predicted = this.predict();
                int lastPredicted = predicted.get(predicted.size() - 1).intValue();
                if (goldLabels.size() >= predicted.size() && lastPredicted == goldLabels.get(predicted.size() - 1)) {
                    correct++;
                }
                total++;
                step++;
                wordIdx = lastPredicted;
            } while (wordIdx != this.endIndex);
            if (predicted.size() < goldLabels.size()) {
                total += goldLabels.size() - predicted.size();
            }
        }
        return new ClassificationPerformance((correct + 0.00) / total);
    }

    @Override
    protected ArrayList<Double> getOutputValue() {
        ArrayList<Double> result = new ArrayList<>();
        Tensor output = outputNode.getValue();
        for (int row = 0; row < output.getShape()[0]; row++) {
            double bestVal = Double.MIN_VALUE;
            double bestIdx = -1;
            for (int col = 0; col < output.getShape()[1]; col++) {
                double v = output.getValue(new int[]{row, col});
                if (v > bestVal) {
                    bestVal = v;
                    bestIdx = col;
                }
            }
            result.add(bestIdx);
        }
        return result;
    }
}
