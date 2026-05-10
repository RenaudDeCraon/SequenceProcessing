import ComputationalGraph.*;
import ComputationalGraph.Function.Sigmoid;
import ComputationalGraph.Function.Tanh;
import ComputationalGraph.Initialization.RandomInitialization;
import ComputationalGraph.Loss.CrossEntropyLoss;
import ComputationalGraph.Optimizer.*;
import Dictionary.VectorizedDictionary;
import Dictionary.Word;
import Dictionary.WordComparator;
import SequenceProcessing.Classification.Gpt1;
import SequenceProcessing.Parameters.Gpt1Parameter;
import org.junit.Test;
import Math.Tensor;

import java.util.ArrayList;
import java.util.Arrays;

public class Gpt1Test {

    @Test
    public void testInitialization() {
        // Create some small dummy training data.
        // Each tensor is a flat sequence where every (d+1)-th value is the class label
        // and the rest are embedding values. Here d=3, so format is [e1, e2, e3, label, ...].
        ArrayList<Tensor> tensors = new ArrayList<>();
        tensors.add(new Tensor(Arrays.asList(
                0.27, 0.67, 0.41, 1.0,
                0.37, 0.17, 0.41, 6.0,
                0.17, 0.65, 0.87, 5.0,
                0.97, 0.19, 0.51, 4.0
        ), new int[]{16}));
        tensors.add(new Tensor(Arrays.asList(
                0.27, 0.67, 0.41, 1.0,
                0.37, 0.17, 0.41, 6.0,
                0.77, 0.61, 0.27, 2.0
        ), new int[]{12}));
        tensors.add(new Tensor(Arrays.asList(
                1.2, 3.6, 7.1, 3.0,
                5.4, 0.17, 9.8, 4.0,
                0.77, 0.61, 0.27, 2.0
        ), new int[]{12}));

        // Feed forward network config: two hidden layers of size 30 and 15
        ArrayList<Integer> hiddenLayers = new ArrayList<>();
        hiddenLayers.add(30);
        hiddenLayers.add(15);
        ArrayList<Object> activationFunctions = new ArrayList<>();
        activationFunctions.add(new Tanh());
        activationFunctions.add(new Sigmoid());

        // We have 2 layer norms in the model (one after attention, one after FFN)
        ArrayList<Double> gamma = new ArrayList<>();
        gamma.add(1.0);
        gamma.add(1.0);
        ArrayList<Double> beta = new ArrayList<>();
        beta.add(0.0);
        beta.add(0.0);

        // Build and train the model
        // wordEmbeddingLength=3, heads=2, vocabSize=7
        ComputationalGraph gpt1 = new Gpt1(new Gpt1Parameter(1, 150,
                new AdamW(0.025, 0.99, 0.99, 0.999, 1e-10, 0.1),
                new RandomInitialization(), new CrossEntropyLoss(),
                3, 2, 7, 1e-9,
                hiddenLayers, activationFunctions,
                gamma, beta),
                new VectorizedDictionary(new WordComparator() {
                    @Override
                    public int compare(Word word, Word word1) {
                        return 0;
                    }
                }));
        gpt1.train(tensors);
    }
}
