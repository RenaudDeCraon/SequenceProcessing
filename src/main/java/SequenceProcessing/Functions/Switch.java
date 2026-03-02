package SequenceProcessing.Functions;

import java.io.Serializable;
import java.util.ArrayList;

import ComputationalGraph.Node.ComputationalNode;
import ComputationalGraph.Node.FunctionNode;
import Math.Tensor;

public class Switch implements ComputationalGraph.Function.Function, Serializable {

    private boolean turn;

    public Switch() {
        this.turn = true;
    }

    public void setTurn(boolean turn) {
        this.turn = turn;
    }

    @Override
    public Tensor calculate(Tensor matrix) {
        if (this.turn) {
            return matrix;
        }
        ArrayList<Double> values = new ArrayList<>();
        int size = 1;
        for (int i = 0; i < matrix.getShape().length; i++) {
            size *= matrix.getShape()[i];
        }
        for (int i = 0; i < size; i++) {
            values.add(0.0);
        }
        return new Tensor(values, matrix.getShape());
    }

    @Override
    public Tensor derivative(Tensor value, Tensor backward) {
        if (this.turn) {
            return backward;
        }
        return calculate(value);
    }

    @Override
    public ComputationalNode addEdge(ArrayList<ComputationalNode> inputNodes, boolean isBiased) {
        ComputationalNode newNode = new FunctionNode(isBiased, this);
        inputNodes.get(0).add(newNode);
        return newNode;
    }
}
