package de.jungblut.online.regularization;

import de.jungblut.math.DoubleVector;

public class GradientDescentUpdater implements WeightUpdater {

  /**
   * Simplistic gradient descent without regularization.
   */
  @Override
  public CostWeightTuple computeNewWeights(DoubleVector theta,
      DoubleVector gradient, double learningRate, long iteration,
      double lambda, double cost) {
    return new CostWeightTuple(cost, theta.subtract(gradient
        .multiply(learningRate)));
  }

}
