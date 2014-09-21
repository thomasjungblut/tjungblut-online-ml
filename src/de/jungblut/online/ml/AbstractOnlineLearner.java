package de.jungblut.online.ml;

import java.util.Optional;
import java.util.function.Supplier;
import java.util.stream.Stream;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;

public abstract class AbstractOnlineLearner<M extends Model> implements
    OnlineLearner<M> {

  protected Supplier<Stream<FeatureOutcomePair>> streamSupplier;
  protected boolean verbose;
  protected int featureDimension;
  protected int outcomeDimension;
  protected int numOutcomeClasses;

  protected void init(Supplier<Stream<FeatureOutcomePair>> streamSupplier) {
    this.streamSupplier = streamSupplier;
    peekDimensions(this.streamSupplier);
  }

  /**
   * Peeks for the feature and outcome dimensions.
   * 
   * @param streamSupplier the supplier that gets streams.
   */
  @VisibleForTesting
  protected void peekDimensions(
      Supplier<Stream<FeatureOutcomePair>> streamSupplier) {
    Stream<FeatureOutcomePair> stream = Preconditions.checkNotNull(
        streamSupplier.get(), "Supplied a null stream!");
    Optional<FeatureOutcomePair> first = stream.findFirst();

    if (!first.isPresent()) {
      throw new IllegalArgumentException("Supplied an empty stream!");
    }

    FeatureOutcomePair firstExample = first.get();
    this.featureDimension = firstExample.getFeature().getDimension();
    this.outcomeDimension = firstExample.getOutcome().getDimension();
    this.numOutcomeClasses = Math.max(2, this.outcomeDimension);
  }

  public void verbose() {
    this.verbose = true;
  }

  public void setVerbose(boolean verbose) {
    this.verbose = verbose;
  }

}
