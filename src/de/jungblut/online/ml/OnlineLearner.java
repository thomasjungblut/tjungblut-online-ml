package de.jungblut.online.ml;

import java.util.function.Supplier;
import java.util.stream.Stream;

/**
 * OnlineLearning interface.
 * 
 * @author thomas.jungblut
 *
 */
public interface OnlineLearner<M extends Model> {

  /**
   * Trains a new model using the supplied streams. In case an algorithm needs
   * multiple iterations, it simply gets a new one from the supplier.
   * 
   * @param streamSupplier the supplier that creates a new stream that can be
   *          consumed.
   * @return the trained model.
   */
  public M train(Supplier<Stream<FeatureOutcomePair>> streamSupplier);

}
