package de.jungblut.online.regularization;

import java.util.function.Function;

import org.junit.Assert;
import org.junit.Test;

import de.jungblut.math.DoubleVector;
import de.jungblut.math.dense.DenseDoubleVector;
import de.jungblut.math.sparse.SequentialSparseDoubleVector;
import de.jungblut.math.sparse.SparseDoubleVector;

public class TestL1Regularizer {

  @Test
  public void testGradientUpdate() {
    WeightUpdater updater = new L1Regularizer(1d, 0d);

    DoubleVector theta = new DenseDoubleVector(new double[] { 1d, 1d, 1d });
    DoubleVector grad = new DenseDoubleVector(new double[] { 1d, 1d, 1d });
    double learningRate = 0.1d;
    CostWeightTuple update = updater.computeNewWeights(theta, grad,
        learningRate, 1, 1d);

    double[] expected = new double[] { 0.9, 0.8, 0.8 };
    Assert.assertArrayEquals(expected, update.getWeight().toArray(), 1e-8);
    Assert.assertEquals(2.8d, update.getCost(), 1e-8);
  }

  @Test
  public void testNoOpUpdate() {
    WeightUpdater updater = new L1Regularizer(0d, 0d);

    DoubleVector theta = new DenseDoubleVector(new double[] { 1d, 1d, 1d });
    DoubleVector grad = new DenseDoubleVector(new double[] { 1d, 1d, 1d });
    double learningRate = 0.1d;
    CostWeightTuple update = updater.computeNewWeights(theta, grad,
        learningRate, 1, 1d);

    Assert.assertArrayEquals(theta.subtract(grad.multiply(learningRate))
        .toArray(), update.getWeight().toArray(), 1e-8);
    Assert.assertEquals(1d, update.getCost(), 0d);

  }

  @Test
  public void testToleranceRemovalDense() {
    baseToleranceRemoval((vec) -> new DenseDoubleVector(vec));
  }

  @Test
  public void testToleranceRemovalSeqSparse() {
    baseToleranceRemoval((vec) -> new SequentialSparseDoubleVector(vec));
  }

  @Test
  public void testToleranceRemovalSparse() {
    baseToleranceRemoval((vec) -> new SparseDoubleVector(vec));
  }

  public void baseToleranceRemoval(
      Function<double[], DoubleVector> vectorFactory) {
    WeightUpdater updater = new L1Regularizer(1d, 0.75);
    DoubleVector theta = vectorFactory.apply(new double[] { 1d, 1d, 1d });
    DoubleVector grad = vectorFactory.apply(new double[] { 1d, 1d, 2d });

    double learningRate = 0.1d;
    CostWeightTuple update = updater.computeNewWeights(theta, grad,
        learningRate, 1, 1d);

    double[] expected = new double[] { 0.9, 0.8, 0 };
    Assert.assertArrayEquals(expected, update.getWeight().toArray(), 1e-8);
    Assert.assertEquals(2.7d, update.getCost(), 1e-8);
  }

  @Test
  public void testSparseVectors() {
    WeightUpdater updater = new L1Regularizer(1d, 0d);

    DoubleVector theta = new SequentialSparseDoubleVector(new double[] { 1d,
        1d, 1d });
    DoubleVector grad = new SequentialSparseDoubleVector(new double[] { 1d, 1d,
        1d });
    double learningRate = 0.1d;
    CostWeightTuple update = updater.computeNewWeights(theta, grad,
        learningRate, 1, 1d);

    double[] expected = new double[] { 0.9, 0.8, 0.8 };
    Assert.assertArrayEquals(expected, update.getWeight().toArray(), 1e-8);
    Assert.assertEquals(2.8d, update.getCost(), 1e-8);
  }

}
