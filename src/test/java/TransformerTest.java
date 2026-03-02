import ComputationalGraph.*;
import ComputationalGraph.Function.Function;
import ComputationalGraph.Function.Sigmoid;
import ComputationalGraph.Function.Tanh;
import ComputationalGraph.Initialization.RandomInitialization;
import ComputationalGraph.Optimizer.*;
import Dictionary.VectorizedDictionary;
import Dictionary.Word;
import Dictionary.WordComparator;
import SequenceProcessing.Classification.Transformer;
import SequenceProcessing.Parameters.TransformerParameter;
import org.junit.Test;
import Math.Tensor;

import java.util.ArrayList;
import java.util.Arrays;

public class TransformerTest {

    @Test
    public void testInitialization() {
        ComputationalGraph transformer = new Transformer(new VectorizedDictionary(new WordComparator() {
            @Override
            public int compare(Word word, Word word1) {
                return 0;
            }
        }));
        ArrayList<Tensor> tensors = new ArrayList<>();
        tensors.add(new Tensor(Arrays.asList(0.2, 0.7, 0.1, 0.3, 0.4, 0.8, 0.9, 0.35, 0.12, 0.27, 0.17, 0.41, Double.MAX_VALUE, 0.27, 0.67, 0.41, 1, 0.37, 0.17, 0.41, 6, 0.17, 0.65, 0.87, 5, 0.97, 0.19, 0.51, 4), new int[]{29}));
        tensors.add(new Tensor(Arrays.asList(0.2, 0.7, 0.1, 0.3, 0.4, 0.8, 0.9, 0.35, 0.12, 0.27, 0.17, 0.41, Double.MAX_VALUE, 0.27, 0.67, 0.41, 1, 0.37, 0.17, 0.41, 6, 0.77, 0.61, 0.27, 2), new int[]{25}));
        tensors.add(new Tensor(Arrays.asList(0.2, 0.7, 0.1, 0.3, 0.4, 0.8, 0.9, 0.35, 0.12, 0.27, 0.17, 0.41, Double.MAX_VALUE, 1.2, 3.6, 7.1, 3, 5.4, 0.17, 9.8, 4, 0.77, 0.61, 0.27, 2), new int[]{25}));
        ArrayList<Integer> input = new ArrayList<>();
        input.add(30);
        input.add(15);
        ArrayList<Function> inputFunctions = new ArrayList<>();
        inputFunctions.add(new Tanh());
        inputFunctions.add(new Sigmoid());
        ArrayList<Function> outputFunctions = new ArrayList<>();
        outputFunctions.add(new Sigmoid());
        outputFunctions.add(new Tanh());
        ArrayList<Double> gammaInput = new ArrayList<>();
        ArrayList<Double> gammaOutput = new ArrayList<>();
        gammaInput.add(1.0);
        gammaInput.add(1.0);
        gammaOutput.add(1.0);
        gammaOutput.add(1.0);
        gammaOutput.add(1.0);
        ArrayList<Double> betaInput = new ArrayList<>();
        ArrayList<Double> betaOutput = new ArrayList<>();
        betaInput.add(0.0);
        betaInput.add(0.0);
        betaOutput.add(0.0);
        betaOutput.add(0.0);
        betaOutput.add(0.0);
        transformer.train(tensors, new TransformerParameter(1, 150, new AdamW(0.025, 0.99, 0.99, 0.999, 1e-10, 0.1), new RandomInitialization(), 3, 2, 7, 1e-9, input, input, inputFunctions, outputFunctions, gammaInput, gammaOutput, betaInput, betaOutput));
    }
}
