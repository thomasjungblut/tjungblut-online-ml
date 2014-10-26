package de.jungblut.online.regularization;

import de.jungblut.math.DoubleVector;

public interface WeightUpdater {

  /**
   * Computes the update for the given weights.
   * 
   * @param theta the old weights.
   * @param gradient the gradient.
   * @param learningRate the learning rate.
   * @param iteration the number of the current iteration.
   * @param lambda the regularization parameter.
   * @param cost the computed cost for this gradient update.
   * @return the gradient vector that should be substracted from the weights and
   *         the updated cost.
   */
  public CostWeightTuple computeNewWeights(DoubleVector theta,
      DoubleVector gradient, double learningRate, long iteration,
      double lambda, double cost);

}
