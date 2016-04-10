package de.jungblut.online.regularization;

import java.util.Iterator;

import org.apache.commons.math3.util.FastMath;

import de.jungblut.math.DoubleVector;
import de.jungblut.math.DoubleVector.DoubleVectorElement;
import de.jungblut.math.minimize.CostGradientTuple;
import de.jungblut.online.ml.FeatureOutcomePair;

/**
 * Based on the paper:
 * http://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf
 * 
 * @author thomas.jungblut
 *
 */
public final class AdaptiveFTRLRegularizer implements WeightUpdater {

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
  public DoubleVector prePredictionWeightUpdate(
      FeatureOutcomePair featureOutcome, DoubleVector theta,
      double learningRate, long iteration) {

    if (squaredPreviousGradient == null) {
      // initialize zeroed vectors of the same type as the weights
      squaredPreviousGradient = theta.deepCopy().multiply(0);
      perCoordinateWeights = theta.deepCopy().multiply(0);
    }

    Iterator<DoubleVectorElement> iterateNonZero = featureOutcome.getFeature()
        .iterateNonZero();
    while (iterateNonZero.hasNext()) {
      DoubleVectorElement next = iterateNonZero.next();
      double gradientValue = next.getValue();
      int index = next.getIndex();

      double zi = perCoordinateWeights.get(index);
      double ni = squaredPreviousGradient.get(index);
      if (FastMath.abs(zi) <= l1) {
        theta.set(index, 0);
      } else {
        double value = -1d / (((beta + FastMath.sqrt(ni)) / learningRate) + l2);
        value = value * (zi - FastMath.signum(gradientValue) * l1);
        theta.set(index, value);
      }
    }

    return theta;
  }

  @Override
  public CostWeightTuple computeNewWeights(DoubleVector theta,
      DoubleVector gradient, double learningRate, long iteration, double cost) {

    Iterator<DoubleVectorElement> iterateNonZero = gradient.iterateNonZero();
    while (iterateNonZero.hasNext()) {
      DoubleVectorElement next = iterateNonZero.next();
      double gradientValue = next.getValue();
      int index = next.getIndex();
      double zi = perCoordinateWeights.get(index);
      double ni = squaredPreviousGradient.get(index);
      // update our cached copies
      double sigma = (FastMath.sqrt(ni + gradientValue * gradientValue) - FastMath
          .sqrt(ni)) / learningRate;
      perCoordinateWeights.set(index,
          zi + gradientValue - sigma * theta.get(index));
      squaredPreviousGradient.set(index, ni + gradientValue * gradientValue);
    }
    return new CostWeightTuple(cost, theta);
  }

  @Override
  public CostGradientTuple updateGradient(DoubleVector theta,
      DoubleVector gradient, double learningRate, long iteration, double cost) {
    return null;
  }

}
