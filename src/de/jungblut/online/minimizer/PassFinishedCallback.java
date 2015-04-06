package de.jungblut.online.minimizer;

import de.jungblut.math.DoubleVector;

public interface PassFinishedCallback {

  /**
   * This callback when a pass over a stream of a minimization objective is
   * finished.
   * 
   * @param pass the number of the current pass.
   * @param iteration the number of the current iteration.
   * @param cost the validation error after the current pass. If no hold-out
   *          validation was chosen, it will be zero.
   * @param currentWeights the current optimal weights.
   * @return false if we should stop the whole computation after this pass, or
   *         true if continue.
   */
  public boolean onPassFinished(int pass, long iteration, double cost,
      DoubleVector currentWeights);

}
