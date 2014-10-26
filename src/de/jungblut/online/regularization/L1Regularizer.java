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
public class L1Regularizer implements WeightUpdater {

  @Override
  public CostWeightTuple computeNewWeights(DoubleVector theta,
      DoubleVector gradient, double learningRate, long iteration,
      double lambda, double cost) {

    if (lambda == 0d) {
      // do simple gradient descent step in this case
      return new CostWeightTuple(cost, theta.subtract(gradient
          .multiply(learningRate)));
    }

    double currentStep = iteration == 0 ? learningRate : learningRate
        / FastMath.sqrt(iteration);
    DoubleVector newWeights = theta.subtract(gradient.multiply(currentStep));
    double shrinkageVal = lambda * currentStep;

    for (int i = 0; i < newWeights.getDimension(); i++) {
      double weight = newWeights.get(i);
      newWeights.set(
          i,
          FastMath.signum(weight)
              * FastMath.max(0.0, FastMath.abs(weight) - shrinkageVal));
    }

    cost += newWeights.abs().sum() * lambda;

    return new CostWeightTuple(cost, newWeights);
  }
}
