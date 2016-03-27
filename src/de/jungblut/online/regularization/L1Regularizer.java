package de.jungblut.online.regularization;

import java.util.Iterator;

import org.apache.commons.math3.util.FastMath;

import de.jungblut.math.DoubleVector;
import de.jungblut.math.DoubleVector.DoubleVectorElement;

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
public final class L1Regularizer extends GradientDescentUpdater {

  private final double tol;
  private final double l1;

  public L1Regularizer(double l1) {
    this.l1 = l1;
    this.tol = l1;
  }

  public L1Regularizer(double l1, double tol) {
    this.l1 = l1;
    this.tol = tol;
  }

  @Override
  public CostWeightTuple computeNewWeights(DoubleVector theta,
      DoubleVector gradient, double learningRate, long iteration, double cost) {

    if (l1 == 0d) {
      // do simple gradient descent step in this case
      return super.computeNewWeights(theta, gradient, learningRate, iteration,
          cost);
    }

    DoubleVector newWeights = theta.subtract(gradient.multiply(learningRate));
    double shrinkageVal = l1 * learningRate;

    double addedCost = 0d;
    if (newWeights.isSparse()) {
      DoubleVector deepCopy = newWeights.deepCopy();
      Iterator<DoubleVectorElement> iterateNonZero = newWeights
          .iterateNonZero();
      while (iterateNonZero.hasNext()) {
        DoubleVectorElement next = iterateNonZero.next();
        if (next.getIndex() > 0) {
          addedCost += updateWeight(newWeights, deepCopy, shrinkageVal,
              next.getIndex(), next.getValue());
        }
      }

      newWeights = deepCopy;
    } else {
      for (int i = 1; i < newWeights.getDimension(); i++) {
        addedCost += updateWeight(newWeights, newWeights, shrinkageVal, i,
            newWeights.get(i));
      }
    }

    cost += addedCost * l1;

    return new CostWeightTuple(cost, newWeights);
  }

  private double updateWeight(DoubleVector newWeights,
      DoubleVector toBeUpdated, double shrinkageVal, int i, double weight) {
    double absWeight = FastMath.abs(weight);
    double newWeight = FastMath.signum(weight)
        * FastMath.max(0.0, absWeight - shrinkageVal);

    if (FastMath.abs(newWeight) < tol) {
      newWeight = 0;
    }

    toBeUpdated.set(i, newWeight);
    return absWeight;
  }

}
