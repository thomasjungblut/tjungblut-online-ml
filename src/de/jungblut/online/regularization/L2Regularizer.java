package de.jungblut.online.regularization;

import de.jungblut.math.DoubleVector;
import de.jungblut.math.minimize.CostGradientTuple;

/**
 * Computes the L2 regularized update: R(w) = (||w||^2) / 2. It assumes the bias
 * feature to be on the very first dimension (zero index) in order to
 * deliberately not regularize it.
 * 
 * @author thomas.jungblut
 *
 */
public final class L2Regularizer extends GradientDescentUpdater {

  private final double l2;

  public L2Regularizer(double l2) {
    this.l2 = l2;
  }

  @Override
  public CostGradientTuple computeGradient(DoubleVector weights,
      DoubleVector gradient, double learningRate, long iteration, double cost) {
    if (l2 != 0d) {
      DoubleVector powered = weights.pow(2d);
      DoubleVector regGrad = weights.multiply(l2);
      // assume bias is on the first dimension
      powered.set(0, 0);
      regGrad.set(0, 0);
      cost += l2 * powered.sum() / 2d;
      gradient = gradient.add(regGrad);
    }
    return new CostGradientTuple(cost, gradient);
  }
}
