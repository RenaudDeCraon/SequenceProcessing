package SequenceProcessing.Classification;

import Classification.Parameter.Parameter;
import Classification.Performance.ClassificationPerformance;
import ComputationalGraph.*;
import Dictionary.*;
import Math.Tensor;
import Math.Vector;
import SequenceProcessing.Functions.*;
import SequenceProcessing.Parameters.TransformerParameter;

import java.io.Serializable;
import java.util.*;

public class Transformer extends ComputationalGraph implements Serializable {

    private final VectorizedDictionary dictionary;
    private int startIndex;
    private final ArrayList<ComputationalNode> gammasInput;
    private final ArrayList<ComputationalNode> gammasOutput;
    private final ArrayList<ComputationalNode> betasInput;
    private final ArrayList<ComputationalNode> betasOutput;

    public Transformer(VectorizedDictionary dictionary) {
        this.dictionary = dictionary;
        this.gammasInput = new ArrayList<>();
        this.betasInput = new ArrayList<>();
        this.gammasOutput = new ArrayList<>();
        this.betasOutput = new ArrayList<>();
        for (int k = 0; k < this.dictionary.size(); k++) {
            if (this.dictionary.getWord(k).getName().equals("<SOS>")) {
                this.startIndex = k;
                break;
            }
        }
    }

    private Tensor positionalEncoding(Tensor tensor, int wordEmbeddingLength) {
        ArrayList<Double> values = new ArrayList<>();
        for (int i = 0; i < tensor.getShape()[1]; i++) {
            for (int j = 0; j < tensor.getShape()[2]; j++) {
                double val = tensor.getValue(new int[]{0, i, j});
                if (j % 2 == 0) {
                    values.add(val + Math.sin((i + 1.0) / Math.pow(10000, (j + 0.0) / wordEmbeddingLength)));
                } else {
                    values.add(val + Math.cos((i + 1.0) / Math.pow(10000, (j - 1.0) / wordEmbeddingLength)));
                }
            }
        }
        return new Tensor(values, tensor.getShape());
    }

    private void createInputTensors(Tensor instance, ComputationalNode input1, ComputationalNode input2, ArrayList<Integer> classLabels, int wordEmbeddingLength) {
        boolean isOutput = false;
        int curLength = 0;
        ArrayList<Double> values = new ArrayList<>();
        for (int i = 0; i < instance.getShape()[0]; i++) {
            double val = instance.getValue(new int[]{i});
            if (val == Double.MAX_VALUE) {
                isOutput = true;
                input1.setValue(new Tensor(values, new int[]{1, curLength / wordEmbeddingLength, wordEmbeddingLength}));
                input1.setValue(positionalEncoding(input1.getValue(), wordEmbeddingLength));
                curLength = 0;
                values.clear();
            } else if (isOutput) {
                if ((curLength + 1) % (wordEmbeddingLength + 1) == 0) {
                    classLabels.add((int) val);
                } else {
                    values.add(val);
                }
                curLength++;
            } else {
                values.add(val);
                curLength++;
            }
        }
        input2.setValue(new Tensor(values, new int[]{1, values.size() / wordEmbeddingLength, wordEmbeddingLength}));
        input2.setValue(positionalEncoding(input2.getValue(), wordEmbeddingLength));
    }

    private ComputationalNode layerNormalization(ComputationalNode input, TransformerParameter parameter, boolean isInput) {
        ArrayList<Double> data = new ArrayList<>();
        ComputationalNode inputC1Mean = this.addEdge(input, new Mean(), false);
        ComputationalNode mean1Minus = this.addEdge(inputC1Mean, new Subtract(), false);
        ComputationalNode inputC1Mean1Minus = this.addAdditionEdge(input, mean1Minus, false);
        ComputationalNode variance1 = this.addEdge(inputC1Mean1Minus, new Variance(), false);
        ComputationalNode rootVariance1 = this.addEdge(variance1, new SquareRoot(parameter.getEpsilon()), false);
        ComputationalNode inverseRootVariance1 = this.addEdge(rootVariance1, new Inverse(), false);
        ComputationalNode lnValue1 = this.addEdge(inputC1Mean1Minus, inverseRootVariance1, false, true);
        if (isInput) {
            for (int j = 0; j < parameter.getL(); j++) {
                data.add(parameter.getGammaInputValue(this.gammasInput.size()));
            }
        } else {
            for (int j = 0; j < parameter.getL(); j++) {
                data.add(parameter.getGammaOutputValue(this.gammasOutput.size()));
            }
        }
        ComputationalNode gammaInput1 = new MultiplicationNode(true, false, new Tensor(data, new int[]{1, 1, parameter.getL()}), true);
        if (isInput) {
            this.gammasInput.add(gammaInput1);
        } else {
            this.gammasOutput.add(gammaInput1);
        }
        ComputationalNode lnValue1GammaInput1 = this.addEdge(lnValue1, gammaInput1, false);
        data.clear();
        if (isInput) {
            for (int j = 0; j < parameter.getL(); j++) {
                data.add(parameter.getBetaInputValue(this.betasInput.size()));
            }
        } else {
            for (int j = 0; j < parameter.getL(); j++) {
                data.add(parameter.getBetaOutputValue(this.betasOutput.size()));
            }
        }
        ComputationalNode betaInput1 = new ComputationalNode(true, false, null, new Tensor(data, new int[]{1, 1, parameter.getL()}));
        if (isInput) {
            this.betasInput.add(betaInput1);
        } else {
            this.betasOutput.add(betaInput1);
        }
        return this.addAdditionEdge(lnValue1GammaInput1, betaInput1, false);
    }

    private ArrayList<ComputationalNode> multiHeadAttention(ComputationalNode input, TransformerParameter parameter, boolean isMasked, Random random) {
        ArrayList<ComputationalNode> nodes = new ArrayList<>();
        ArrayList<Double> data = new ArrayList<>();
        for (int i = 0; i < parameter.getN(); i++) {
            data.clear();
            for (int j = 0; j < parameter.getL() * parameter.getDk(); j++) {
                data.add(-0.01 + (0.02 * random.nextDouble()));
            }
            ComputationalNode wk = new MultiplicationNode(true, false, new Tensor(data, new int[]{1, parameter.getL(), parameter.getDk()}), false);
            ComputationalNode k = this.addEdge(input, wk, false);
            data.clear();
            for (int j = 0; j < parameter.getL() * parameter.getDk(); j++) {
                data.add(-0.01 + (0.02 * random.nextDouble()));
            }
            ComputationalNode wq = new MultiplicationNode(true, false, new Tensor(data, new int[]{1, parameter.getL(), parameter.getDk()}), false);
            ComputationalNode q = this.addEdge(input, wq, false);
            data.clear();
            for (int j = 0; j < parameter.getL() * parameter.getDk(); j++) {
                data.add(-0.01 + (0.02 * random.nextDouble()));
            }
            ComputationalNode wv = new MultiplicationNode(true, false, new Tensor(data, new int[]{1, parameter.getL(), parameter.getDk()}), false);
            ComputationalNode v = this.addEdge(input, wv, false);
            ComputationalNode kTranspose = this.addEdge(k, new Transpose(), false);
            ComputationalNode qk = this.addEdge(q, kTranspose, false, false);
            ComputationalNode qkDk = this.addEdge(qk, new MultiplyByConstant(1.0 / Math.sqrt(parameter.getDk())), false);
            ComputationalNode sQkDk;
            if (isMasked) {
                ComputationalNode mQkDk = this.addEdge(qkDk, new Mask(), false);
                sQkDk = this.addEdge(mQkDk, new Softmax(), false);
            } else {
                sQkDk = this.addEdge(qkDk, new Softmax(), false);
            }
            ComputationalNode attention = this.addEdge(sQkDk, v, false);
            nodes.add(attention);
        }
        return nodes;
    }

    private ComputationalNode feedforwardNeuralNetwork(ComputationalNode current, int currentLayerSize, TransformerParameter parameter, Random random, boolean isInput) {
        int size;
        if (isInput) {
            size = parameter.getInputSize();
        } else {
            size = parameter.getOutputSize();
        }
        ArrayList<Double> data = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            if (isInput) {
                data.clear();
                for (int j = 0; j < currentLayerSize * parameter.getInputHiddenLayer(i); j++) {
                    data.add(-0.01 + (0.02 * random.nextDouble()));
                }
                ComputationalNode hiddenWeight = new MultiplicationNode(true, false, new Tensor(data, new int[]{1, currentLayerSize, parameter.getInputHiddenLayer(i)}), false);
                ComputationalNode hiddenLayer = this.addEdge(current, hiddenWeight, false);
                current = this.addEdge(hiddenLayer, parameter.getInputActivationFunction(i), true);
                currentLayerSize = parameter.getInputHiddenLayer(i) + 1;
            } else {
                data.clear();
                for (int j = 0; j < currentLayerSize * parameter.getOutputHiddenLayer(i); j++) {
                    data.add(-0.01 + (0.02 * random.nextDouble()));
                }
                ComputationalNode hiddenWeight = new MultiplicationNode(true, false, new Tensor(data, new int[]{1, currentLayerSize, parameter.getOutputHiddenLayer(i)}), false);
                ComputationalNode hiddenLayer = this.addEdge(current, hiddenWeight, false);
                current = this.addEdge(hiddenLayer, parameter.getOutputActivationFunction(i), true);
                currentLayerSize = parameter.getOutputHiddenLayer(i) + 1;
            }
        }
        data.clear();
        for (int j = 0; j < currentLayerSize * parameter.getL(); j++) {
            data.add(-0.01 + (0.02 * random.nextDouble()));
        }
        ComputationalNode outputWeight = new MultiplicationNode(true, false, new Tensor(data, new int[]{1, currentLayerSize, parameter.getL()}), false);
        ComputationalNode outputLayer = this.addEdge(current, outputWeight, false);
        return this.addEdge(outputLayer, new Softmax(), false);
    }

    @Override
    public void train(ArrayList<Tensor> trainSet, Parameter parameter) {
        ArrayList<Double> data = new ArrayList<>();
        Random random = new Random(parameter.getSeed());
        // Encoder Block
        ComputationalNode input1 = new MultiplicationNode(false, true, false);
        this.inputNodes.add(input1);
        ConcatenatedNode concatenatedNode1 = (ConcatenatedNode) this.concatEdges(multiHeadAttention(input1, ((TransformerParameter) parameter), false, random), 2);
        for (int j = 0; j < ((TransformerParameter) parameter).getL() * ((TransformerParameter) parameter).getL(); j++) {
            data.add(-0.01 + (0.02 * random.nextDouble()));
        }
        ComputationalNode we = new MultiplicationNode(true, false, new Tensor(data, new int[]{1, ((TransformerParameter) parameter).getL(), ((TransformerParameter) parameter).getL()}), false);
        ComputationalNode c1 = this.addEdge(concatenatedNode1, we, false);
        ComputationalNode inputC1 = this.addAdditionEdge(input1, c1, false);
        ComputationalNode y1 = layerNormalization(inputC1, ((TransformerParameter) parameter), true);
        ComputationalNode oe = this.addAdditionEdge(feedforwardNeuralNetwork(y1, ((TransformerParameter) parameter).getL(), ((TransformerParameter) parameter), random, true), y1, false);
        ComputationalNode encoder = layerNormalization(oe, ((TransformerParameter) parameter), true);
        // Decoder Block
        ComputationalNode input2 = new MultiplicationNode(false, true, false);
        this.inputNodes.add(input2);
        ConcatenatedNode concatenatedNode2 = (ConcatenatedNode) this.concatEdges(multiHeadAttention(input2, ((TransformerParameter) parameter), true, random), 2);
        data.clear();
        for (int j = 0; j < ((TransformerParameter) parameter).getL() * ((TransformerParameter) parameter).getL(); j++) {
            data.add(-0.01 + (0.02 * random.nextDouble()));
        }
        ComputationalNode wd1 = new MultiplicationNode(true, false, new Tensor(data, new int[]{1, ((TransformerParameter) parameter).getL(), ((TransformerParameter) parameter).getL()}), false);
        ComputationalNode c2 = this.addEdge(concatenatedNode2, wd1, false);
        ComputationalNode inputC2 = this.addAdditionEdge(input2, c2, false);
        ComputationalNode cd2 = layerNormalization(inputC2, ((TransformerParameter) parameter), false);
        ArrayList<ComputationalNode> nodes = new ArrayList<>();
        for (int i = 0; i < ((TransformerParameter) parameter).getN(); i++) {
            data.clear();
            for (int j = 0; j < ((TransformerParameter) parameter).getL() * ((TransformerParameter) parameter).getDk(); j++) {
                data.add(-0.01 + (0.02 * random.nextDouble()));
            }
            ComputationalNode wk = new MultiplicationNode(true, false, new Tensor(data, new int[]{1, ((TransformerParameter) parameter).getL(), ((TransformerParameter) parameter).getDk()}), false);
            ComputationalNode k = this.addEdge(encoder, wk, false);
            data.clear();
            for (int j = 0; j < ((TransformerParameter) parameter).getL() * ((TransformerParameter) parameter).getDk(); j++) {
                data.add(-0.01 + (0.02 * random.nextDouble()));
            }
            ComputationalNode wq = new MultiplicationNode(true, false, new Tensor(data, new int[]{1, ((TransformerParameter) parameter).getL(), ((TransformerParameter) parameter).getDk()}), false);
            ComputationalNode q = this.addEdge(cd2, wq, false);
            data.clear();
            for (int j = 0; j < ((TransformerParameter) parameter).getL() * ((TransformerParameter) parameter).getDk(); j++) {
                data.add(-0.01 + (0.02 * random.nextDouble()));
            }
            ComputationalNode wv = new MultiplicationNode(true, false, new Tensor(data, new int[]{1, ((TransformerParameter) parameter).getL(), ((TransformerParameter) parameter).getDk()}), false);
            ComputationalNode v = this.addEdge(encoder, wv, false);
            ComputationalNode kTranspose = this.addEdge(k, new Transpose(), false);
            ComputationalNode qk = this.addEdge(q, kTranspose, false, false);
            ComputationalNode qkDk = this.addEdge(qk, new MultiplyByConstant(1.0 / Math.sqrt(((TransformerParameter) parameter).getDk())), false);
            ComputationalNode sQkDk = this.addEdge(qkDk, new Softmax(), false);
            ComputationalNode attention = this.addEdge(sQkDk, v, false);
            nodes.add(attention);
        }
        ConcatenatedNode concatenatedNode3 = (ConcatenatedNode) this.concatEdges(nodes, 2);
        data.clear();
        for (int j = 0; j < ((TransformerParameter) parameter).getL() * ((TransformerParameter) parameter).getL(); j++) {
            data.add(-0.01 + (0.02 * random.nextDouble()));
        }
        ComputationalNode wd2 = new MultiplicationNode(true, false, new Tensor(data, new int[]{1, ((TransformerParameter) parameter).getL(), ((TransformerParameter) parameter).getL()}), false);
        ComputationalNode cd3 = this.addEdge(concatenatedNode3, wd2, false);
        ComputationalNode cd3cd2 = this.addAdditionEdge(cd2, cd3, false);
        ComputationalNode yd1 = this.layerNormalization(cd3cd2, ((TransformerParameter) parameter), false);
        ComputationalNode od = this.feedforwardNeuralNetwork(yd1, ((TransformerParameter) parameter).getL(), ((TransformerParameter) parameter), random, false);
        ComputationalNode oy = this.addAdditionEdge(od, yd1, false);
        ComputationalNode d = this.layerNormalization(oy, ((TransformerParameter) parameter), false);
        data.clear();
        for (int j = 0; j < ((TransformerParameter) parameter).getL() * ((TransformerParameter) parameter).getV(); j++) {
            data.add(-0.01 + (0.02 * random.nextDouble()));
        }
        ComputationalNode wdo = new MultiplicationNode(true, false, new Tensor(data, new int[]{1, ((TransformerParameter) parameter).getL(), ((TransformerParameter) parameter).getV()}), false);
        ComputationalNode decoder = this.addEdge(d, wdo, false);
        this.addEdge(decoder, new Softmax(), false);
        // Training
        for (int i = 0; i < ((TransformerParameter) parameter).getEpoch(); i++) {
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
                createInputTensors(instance, this.inputNodes.get(0), this.inputNodes.get(1), classLabels, ((TransformerParameter) parameter).getL() - 1);
                broadcast(true);
                broadcast(false);
                this.forwardCalculation();
                this.backpropagation(((TransformerParameter) parameter).getLearningRate(), classLabels);
                updateGammaAndBetaValues();
                classLabels.clear();
            }
            ((TransformerParameter) parameter).setLearningRate();
        }
    }

    private void updateGammaAndBetaValues() {
        int L = this.gammasInput.get(0).getValue().getShape()[2];
        for (int i = 0; i < this.gammasInput.size(); i++) {
            double[] gammaValues = new double[L];
            double[] betaValues = new double[L];
            for (int j = 0; j < this.gammasInput.get(i).getValue().getShape()[1]; j++) {
                for (int k = 0; k < L; k++) {
                    gammaValues[k] += this.gammasInput.get(i).getValue().getValue(new int[]{0, j, k}) / this.gammasInput.get(i).getValue().getShape()[1];
                    betaValues[k] += this.betasInput.get(i).getValue().getValue(new int[]{0, j, k}) / this.gammasInput.get(i).getValue().getShape()[1];
                }
            }
            ArrayList<Double> gammaValuesList = new ArrayList<>();
            ArrayList<Double> betaValuesList = new ArrayList<>();
            for (int j = 0; j < gammaValues.length; j++) {
                gammaValuesList.add(gammaValues[j]);
                betaValuesList.add(betaValues[j]);
            }
            this.gammasInput.get(i).setValue(new Tensor(gammaValuesList, new int[]{1, 1, L}));
            this.betasInput.get(i).setValue(new Tensor(betaValuesList, new int[]{1, 1, L}));
        }
        for (int i = 0; i < this.gammasOutput.size(); i++) {
            double[] gammaValues = new double[L];
            double[] betaValues = new double[L];
            for (int j = 0; j < this.gammasOutput.get(i).getValue().getShape()[1]; j++) {
                for (int k = 0; k < L; k++) {
                    gammaValues[k] += this.gammasOutput.get(i).getValue().getValue(new int[]{0, j, k}) / this.gammasOutput.get(i).getValue().getShape()[1];
                    betaValues[k] += this.betasOutput.get(i).getValue().getValue(new int[]{0, j, k}) / this.gammasOutput.get(i).getValue().getShape()[1];
                }
            }
            ArrayList<Double> gammaValuesList = new ArrayList<>();
            ArrayList<Double> betaValuesList = new ArrayList<>();
            for (int j = 0; j < gammaValues.length; j++) {
                gammaValuesList.add(gammaValues[j]);
                betaValuesList.add(betaValues[j]);
            }
            this.gammasOutput.get(i).setValue(new Tensor(gammaValuesList, new int[]{1, 1, L}));
            this.betasOutput.get(i).setValue(new Tensor(betaValuesList, new int[]{1, 1, L}));
        }
    }

    private void broadcast(boolean isInput) {
        int row, column;
        ArrayList<ComputationalNode> gammas, betas;
        if (isInput) {
            row = this.inputNodes.get(0).getValue().getShape()[1];
            column = this.inputNodes.get(0).getValue().getShape()[2] + 1;
            gammas = this.gammasInput;
            betas = this.betasInput;
        } else {
            row = this.inputNodes.get(1).getValue().getShape()[1];
            column = this.inputNodes.get(1).getValue().getShape()[2] + 1;
            gammas = this.gammasOutput;
            betas = this.betasOutput;
        }
        for (int k = 0; k < gammas.size(); k++) {
            ArrayList<Double> gammaValues = new ArrayList<>();
            ArrayList<Double> betaValues = new ArrayList<>();
            for (int i = 0; i < row; i++) {
                for (int j = 0; j < column; j++) {
                    gammaValues.add(gammas.get(k).getValue().getValue(new int[]{0, 0, j}));
                    betaValues.add(betas.get(k).getValue().getValue(new int[]{0, 0, j}));
                }
            }
            gammas.get(k).setValue(new Tensor(gammaValues, new int[]{1, row, column}));
            betas.get(k).setValue(new Tensor(betaValues, new int[]{1, row, column}));
        }
    }

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
        node.setValue(new Tensor(data, new int[]{1, bound, vector.size()}));
    }

    @Override
    public ClassificationPerformance test(ArrayList<Tensor> testSet) {
        int count = 0, total = 0;
        for (Tensor instance : testSet) {
            ArrayList<Integer> goldClassLabels = new ArrayList<>(), classLabels;
            createInputTensors(instance, this.inputNodes.get(0), new ComputationalNode(false, false, null, null), goldClassLabels, ((VectorizedWord) this.dictionary.getWord(0)).getVector().size());
            broadcast(true);
            int j = 1;
            int currentWordIndex = this.startIndex;
            do {
                setInputNode(j, ((VectorizedWord) this.dictionary.getWord(currentWordIndex)).getVector(), this.inputNodes.get(1));
                broadcast(false);
                classLabels = this.predict();
                if (classLabels.get(classLabels.size() - 1).equals(goldClassLabels.get(classLabels.size() - 1))) {
                    count++;
                }
                total++;
                j++;
                currentWordIndex = classLabels.get(classLabels.size() - 1);
            } while (classLabels.size() != goldClassLabels.size());
        }
        return new ClassificationPerformance((count + 0.00) / total);
    }

    @Override
    protected ArrayList<Integer> getClassLabels(ComputationalNode computationalNode) {
        ArrayList<Integer> classLabels = new ArrayList<>();
        Tensor value = computationalNode.getValue();
        for (int i = 0; i < value.getShape()[1]; i++) {
            double max = Double.MIN_VALUE;
            int index = -1;
            for (int j = 0; j < value.getShape()[2]; j++) {
                if (value.getValue(new int[]{0, i, j}) > max) {
                    max = value.getValue(new int[]{0, i, j});
                    index = j;
                }
            }
            classLabels.add(index);
        }
        return classLabels;
    }
}
