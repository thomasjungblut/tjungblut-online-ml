package de.jungblut.online.regularization;

import java.util.Iterator;

import org.apache.commons.math3.util.FastMath;

import de.jungblut.math.DoubleVector;
import de.jungblut.math.DoubleVector.DoubleVectorElement;
import de.jungblut.math.sparse.SparseDoubleVector;

/**
 * Based on the paper:
 * http://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf
 * 
 * @author thomas.jungblut
 *
 */
public final class AdaptiveFTRLRegularizer extends GradientDescentUpdater {

  private final double beta;
  private final double l1;
  private final double l2;

  private DoubleVector squaredPreviousGradient; // n in the paper
  private DoubleVector perCoordinateWeights; // z in the paper

  /**
   * Creates a new AdaptiveFTRLRegularizer.
   * 
   * @param beta the smoothing parameter for the learning rate.
   * @param l1 the l1 regularization.
   * @param l2 the l2 regularization.
   */
  public AdaptiveFTRLRegularizer(double beta, double l1, double l2) {
    this.beta = beta;
    this.l1 = l1;
    this.l2 = l2;
  }

  @Override
  public CostWeightTuple computeNewWeights(DoubleVector theta,
      DoubleVector gradient, double learningRate, long iteration, double cost) {

    if (squaredPreviousGradient == null) {
      squaredPreviousGradient = new SparseDoubleVector(theta.getDimension());
      perCoordinateWeights = new SparseDoubleVector(theta.getDimension());
    }

    Iterator<DoubleVectorElement> iterateNonZero = gradient.iterateNonZero();
    while (iterateNonZero.hasNext()) {
      DoubleVectorElement next = iterateNonZero.next();
      double gradientValue = next.getValue();
      int index = next.getIndex();
      double sign = FastMath.signum(gradientValue);

      double zi = perCoordinateWeights.get(index);
      double ni = squaredPreviousGradient.get(index);
      if (sign * zi <= l1) {
        theta.set(index, 0);
      } else {
        double value = (sign * l1 - zi)
            / ((beta + FastMath.sqrt(ni)) / learningRate + l2);
        theta.set(index, value);
      }

      // update our cached copies
      double sigma = (FastMath.sqrt(ni + gradientValue * gradientValue) - FastMath
          .sqrt(ni)) / learningRate;
      perCoordinateWeights.set(index,
          zi + gradientValue - sigma * theta.get(index));
      squaredPreviousGradient.set(index, ni + gradientValue * gradientValue);
    }

    return new CostWeightTuple(cost, theta);
  }
}
