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
        ArrayList<Tensor> data = new ArrayList<>();
        data.add(new Tensor(Arrays.asList(
                0.27, 0.67, 0.41, 1.0,
                0.37, 0.17, 0.41, 6.0,
                0.17, 0.65, 0.87, 5.0,
                0.97, 0.19, 0.51, 4.0
        ), new int[]{16}));
        data.add(new Tensor(Arrays.asList(
                0.27, 0.67, 0.41, 1.0,
                0.37, 0.17, 0.41, 6.0,
                0.77, 0.61, 0.27, 2.0
        ), new int[]{12}));
        data.add(new Tensor(Arrays.asList(
                1.2, 3.6, 7.1, 3.0,
                5.4, 0.17, 9.8, 4.0,
                0.77, 0.61, 0.27, 2.0
        ), new int[]{12}));
        ArrayList<Integer> layers = new ArrayList<>();
        layers.add(30);
        layers.add(15);
        ArrayList<Object> functions = new ArrayList<>();
        functions.add(new Tanh());
        functions.add(new Sigmoid());
        ArrayList<Double> gammaVals = new ArrayList<>();
        gammaVals.add(1.0);
        gammaVals.add(1.0);
        ArrayList<Double> betaVals = new ArrayList<>();
        betaVals.add(0.0);
        betaVals.add(0.0);
        Gpt1Parameter params = new Gpt1Parameter(1, 150,
                new AdamW(0.025, 0.99, 0.99, 0.999, 1e-10, 0.1),
                new RandomInitialization(), new CrossEntropyLoss(),
                3, 2, 7, 1e-9,
                layers, functions, gammaVals, betaVals);
        ComputationalGraph model = new Gpt1(params,
                new VectorizedDictionary(new WordComparator() {
                    @Override
                    public int compare(Word word, Word word1) {
                        return 0;
                    }
                }));
        model.train(data);
    }
}
