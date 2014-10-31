package de.jungblut.online.regularization;

import org.apache.commons.math3.util.FastMath;

import de.jungblut.math.DoubleVector;

/**
 * Ported to "real" Java from Spark's mllib
 * org.apache.spark.mllib.optimization.Updater.
 * 
 * L1 regularizer: R(w) = ||w||_1.
 * 
 * Uses a step-size decreasing with the square root of the number of iterations.
 * 
 * Instead of subgradient of the regularizer, the proximal operator for the L1
 * regularization is applied after the gradient step. This is known to result in
 * better sparsity of the intermediate solution.
 */
public class L1Regularizer extends GradientDescentUpdater {

  @Override
  public CostWeightTuple computeNewWeights(DoubleVector theta,
      DoubleVector gradient, double learningRate, long iteration,
      double lambda, double cost) {

    if (lambda == 0d) {
      // do simple gradient descent step in this case
      return super.computeNewWeights(theta, gradient, learningRate, iteration,
          lambda, cost);
    }

    DoubleVector newWeights = theta.subtract(gradient.multiply(learningRate));
    double shrinkageVal = lambda * learningRate;

    double addedCost = 0d;
    // don't regularize the bias
    for (int i = 1; i < newWeights.getDimension(); i++) {
      double weight = newWeights.get(i);
      double absWeight = FastMath.abs(weight);
      double newWeight = FastMath.signum(weight)
          * FastMath.max(0.0, absWeight - shrinkageVal);
      newWeights.set(i, newWeight);
      addedCost += absWeight;
    }

    cost += addedCost * lambda;

    return new CostWeightTuple(cost, newWeights);
  }
}
