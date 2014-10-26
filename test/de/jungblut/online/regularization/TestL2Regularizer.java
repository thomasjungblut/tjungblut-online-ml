package de.jungblut.online.regularization;

import org.junit.Assert;
import org.junit.Test;

import de.jungblut.math.DoubleVector;
import de.jungblut.math.dense.DenseDoubleVector;

public class TestL2Regularizer {

  @Test
  public void testGradientUpdate() {
    WeightUpdater updater = new L2Regularizer();

    DoubleVector theta = new DenseDoubleVector(new double[] { 1d, 1d, 1d });
    DoubleVector grad = new DenseDoubleVector(new double[] { 1d, 1d, 1d });
    double learningRate = 0.1d;
    CostWeightTuple update = updater.computeNewWeights(theta, grad,
        learningRate, 1, 1d, 1d);

    double[] expected = new double[] { 0.9, 0.8, 0.8 };
    Assert.assertArrayEquals(expected, update.getWeight().toArray(), 1e-8);
    Assert.assertEquals(2d, update.getCost(), 0d);
  }

  @Test
  public void testNoOpUpdate() {
    WeightUpdater updater = new L2Regularizer();

    DoubleVector theta = new DenseDoubleVector(new double[] { 1d, 1d, 1d });
    DoubleVector grad = new DenseDoubleVector(new double[] { 1d, 1d, 1d });
    double learningRate = 0.1d;
    CostWeightTuple update = updater.computeNewWeights(theta, grad,
        learningRate, 1, 0d, 1d);

    Assert.assertArrayEquals(theta.subtract(grad.multiply(learningRate))
        .toArray(), update.getWeight().toArray(), 1e-8);
    Assert.assertEquals(1d, update.getCost(), 0d);

  }

}
