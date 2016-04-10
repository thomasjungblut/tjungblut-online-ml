package de.jungblut.online.regularization;

import de.jungblut.math.DoubleVector;
import de.jungblut.math.minimize.CostGradientTuple;
import de.jungblut.online.ml.FeatureOutcomePair;

// TODO split this into three interfaces
public interface WeightUpdater {

  /**
   * Computes a pre-prediction time weight update.
   * 
   * @param featureOutcome the current feature outcome pair
   * @param theta the weights to augment.
   * @param learningRate the learning rate.
   * @param iteration the number of the current iteration.
   * @return a changed weight vector or just plainly theta.
   */
  public DoubleVector prePredictionWeightUpdate(
      FeatureOutcomePair featureOutcome, DoubleVector theta,
      double learningRate, long iteration);

  /**
   * Computes the update for the given weights.
   * 
   * @param theta the old weights.
   * @param gradient the pre-computed gradient from the loss function.
   * @param learningRate the learning rate.
   * @param iteration the number of the current iteration.
   * @param cost the computed cost for this gradient update.
   * @return the already updated weights for a particular updated gradient.
   */
  public CostWeightTuple computeNewWeights(DoubleVector theta,
      DoubleVector gradient, double learningRate, long iteration, double cost);

  /**
   * Updates the gradient.
   * 
   * @param theta the old weights.
   * @param gradient the pre-computed gradient from the loss function.
   * @param learningRate the learning rate.
   * @param iteration the number of the current iteration.
   * @param cost the computed cost for this gradient update.
   * @return the gradient vector that should be substracted from the weights and
   *         the updated cost.
   */
  public CostGradientTuple updateGradient(DoubleVector theta,
      DoubleVector gradient, double learningRate, long iteration, double cost);

}
