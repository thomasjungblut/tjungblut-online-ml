package de.jungblut.online.regularization;

import de.jungblut.math.DoubleVector;
import de.jungblut.math.minimize.CostGradientTuple;
import de.jungblut.online.ml.FeatureOutcomePair;

public class GradientDescentUpdater implements WeightUpdater {

  /**
   * Simplistic gradient descent without regularization.
   */
  @Override
  public CostWeightTuple computeNewWeights(DoubleVector theta,
      DoubleVector gradient, double learningRate, long iteration, double cost) {

    CostGradientTuple gradientTuple = updateGradient(theta, gradient,
        learningRate, iteration, cost);

    DoubleVector dampened = gradientTuple.getGradient().multiply(learningRate);
    DoubleVector newWeights = theta.subtract(dampened);

    return new CostWeightTuple(gradientTuple.getCost(), newWeights);
  }

  @Override
  public CostGradientTuple updateGradient(DoubleVector theta,
      DoubleVector gradient, double learningRate, long iteration, double cost) {
    return new CostGradientTuple(cost, gradient);
  }

  @Override
  public DoubleVector prePredictionWeightUpdate(
      FeatureOutcomePair featureOutcome, DoubleVector theta,
      double learningRate, long iteration) {
    return theta;
  }
}
