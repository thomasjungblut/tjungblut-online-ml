package de.jungblut.online.regularization;

import de.jungblut.math.DoubleVector;
import de.jungblut.math.minimize.CostGradientTuple;

public interface WeightUpdater {

  /**
   * Computes the update for the given weights.
   * 
   * @param theta the old weights.
   * @param gradient the gradient.
   * @param learningRate the learning rate.
   * @param iteration the number of the current iteration.
   * @param cost the computed cost for this gradient update.
   * @return the already updated weights for a particular updated gradient.
   */
  public CostWeightTuple computeNewWeights(DoubleVector theta,
      DoubleVector gradient, double learningRate, long iteration, double cost);

  /**
   * Computes the gradient.
   * 
   * @param theta the old weights.
   * @param gradient the gradient.
   * @param learningRate the learning rate.
   * @param iteration the number of the current iteration.
   * @param cost the computed cost for this gradient update.
   * @return the gradient vector that should be substracted from the weights and
   *         the updated cost.
   */
  public CostGradientTuple computeGradient(DoubleVector theta,
      DoubleVector gradient, double learningRate, long iteration, double cost);

}
