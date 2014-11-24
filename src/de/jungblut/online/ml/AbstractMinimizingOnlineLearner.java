package de.jungblut.online.ml;

import java.util.Random;
import java.util.function.Supplier;
import java.util.stream.Stream;

import com.google.common.base.Preconditions;

import de.jungblut.math.DoubleVector;
import de.jungblut.math.dense.DenseDoubleVector;
import de.jungblut.math.minimize.CostGradientTuple;
import de.jungblut.math.sparse.SparseDoubleVector;
import de.jungblut.online.minimizer.StochasticMinimizer;

public abstract class AbstractMinimizingOnlineLearner<M extends Model> extends
    AbstractOnlineLearner<M> {

  protected final StochasticMinimizer minimizer;

  protected Random random = new Random();
  protected int numPasses = 1;

  protected boolean sparseWeights;

  public AbstractMinimizingOnlineLearner(StochasticMinimizer minimizer) {
    this.minimizer = minimizer;
  }

  @Override
  public M train(Supplier<Stream<FeatureOutcomePair>> streamSupplier) {

    init(streamSupplier);

    DoubleVector weights = randomInitialize(featureDimension);
    DoubleVector minimized = minimizer.minimize(weights, streamSupplier,
        this::observeExampleSafe, numPasses, verbose);

    return createModel(minimized);
  }

  /**
   * Observes the next example.
   * 
   * @param next the next feature/outcome pair.
   * @param weights the current weights.
   * @return a cost gradient tuple that can be used for minimization.
   */
  protected abstract CostGradientTuple observeExample(FeatureOutcomePair next,
      DoubleVector weights);

  /**
   * Creates a model with the given minimized weights.
   * 
   * @param weights the learned weights.
   * @return a model that describes the weights.
   */
  protected abstract M createModel(DoubleVector weights);

  protected CostGradientTuple observeExampleSafe(FeatureOutcomePair next,
      DoubleVector weights) {
    // do some sanity checks before we actually do the computation
    Preconditions.checkArgument(weights.getDimension() == featureDimension,
        "Feature dimension must match the weight dimension! Expected: "
            + featureDimension + ", given " + weights.getDimension());
    Preconditions.checkArgument(featureDimension == next.getFeature()
        .getDimension(),
        "Feature dimension must match the initially set dimension! Expected: "
            + featureDimension + ", given " + next.getFeature().getDimension());
    Preconditions.checkArgument(outcomeDimension == next.getOutcome()
        .getDimension(),
        "Outcome dimension must match the initially set dimension! Expected: "
            + outcomeDimension + ", given " + next.getOutcome().getDimension());
    return observeExample(next, weights);
  }

  protected DoubleVector randomInitialize(int dimension) {
    if (sparseWeights) {
      return new SparseDoubleVector(dimension);
    } else {
      double[] array = new double[dimension];
      for (int i = 0; i < array.length; i++) {
        array[i] = (random.nextDouble() * 2) - 1d;
      }
      return new DenseDoubleVector(array);
    }
  }

  public void setRandom(Random random) {
    this.random = Preconditions.checkNotNull(random,
        "Supplied random was null!");
  }

  public void useSparseWeights() {
    sparseWeights = true;
  }

  public void setNumPasses(int passes) {
    Preconditions
        .checkArgument(passes > 0,
            "Iterative algorithms need at least a single pass. Supplied: "
                + passes);
    this.numPasses = passes;
  }

}
