package de.jungblut.online.bayes;

import java.util.Iterator;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;
import java.util.stream.Stream;

import org.apache.commons.math3.util.FastMath;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import de.jungblut.math.DoubleMatrix;
import de.jungblut.math.DoubleVector;
import de.jungblut.math.DoubleVector.DoubleVectorElement;
import de.jungblut.math.dense.DenseDoubleVector;
import de.jungblut.math.sparse.SparseDoubleRowMatrix;
import de.jungblut.online.ml.AbstractOnlineLearner;
import de.jungblut.online.ml.FeatureOutcomePair;

/**
 * Multinomial naive bayes learner. This class now contains a sparse internal
 * representations of the "feature given class" probabilities. Thus it can be
 * scaled to very large text corpora and large numbers of classes easily.
 * 
 * The internal accesses are thread-safe, so a parallel stream can be used.
 * 
 * @author thomas.jungblut
 * 
 */
public class NaiveBayesLearner extends
    AbstractOnlineLearner<BayesianProbabilityModel> {

  private static final Logger LOG = LogManager
      .getLogger(NaiveBayesLearner.class);

  private DoubleMatrix probabilityMatrix;
  private DoubleVector classPriorProbability;

  private boolean verbose;

  /**
   * Default constructor to construct this classifier.
   */
  public NaiveBayesLearner() {
  }

  /**
   * Pass true if this classifier should output some progress information to the
   * logger.
   */
  public NaiveBayesLearner(boolean verbose) {
    this.verbose = verbose;
  }

  @Override
  public BayesianProbabilityModel train(
      Supplier<Stream<FeatureOutcomePair>> streamSupplier) {

    init(streamSupplier);

    Stream<FeatureOutcomePair> stream = streamSupplier.get();

    // sparse row representations, so every class has the features as a hashset
    // of values. This gives good compression for many class problems.
    probabilityMatrix = new SparseDoubleRowMatrix(numOutcomeClasses,
        featureDimension);

    int[] tokenPerClass = new int[numOutcomeClasses];
    int[] numDocumentsPerClass = new int[numOutcomeClasses];

    // observe the probabilities
    AtomicInteger numDocumentsSeen = new AtomicInteger(0);
    stream.forEach((pair) -> {
      observe(pair.getFeature(), pair.getOutcome(), numOutcomeClasses,
          tokenPerClass, numDocumentsPerClass);
      numDocumentsSeen.incrementAndGet();
    });

    // know we know the token distribution per class, we can calculate the
    // probability. It is intended for them to be negative in some cases
    for (int row = 0; row < numOutcomeClasses; row++) {
      // we can quite efficiently iterate over the non-zero row vectors now
      DoubleVector rowVector = probabilityMatrix.getRowVector(row);
      // don't care about not occuring words, we honor them with a very small
      // probability later on when predicting, here we save a lot space.
      Iterator<DoubleVectorElement> iterateNonZero = rowVector.iterateNonZero();
      double normalizer = FastMath.log(tokenPerClass[row]
          + probabilityMatrix.getColumnCount() - 1);
      while (iterateNonZero.hasNext()) {
        DoubleVectorElement next = iterateNonZero.next();
        double currentWordCount = next.getValue();
        double logProbability = FastMath.log(currentWordCount) - normalizer;
        probabilityMatrix.set(row, next.getIndex(), logProbability);
      }
      if (verbose) {
        LOG.info("Computed " + row + " / " + numOutcomeClasses + "!");
      }
    }

    classPriorProbability = new DenseDoubleVector(numOutcomeClasses);
    for (int i = 0; i < numOutcomeClasses; i++) {
      double prior = FastMath.log(numDocumentsPerClass[i])
          - FastMath.log(numDocumentsSeen.get());
      classPriorProbability.set(i, prior);
    }
    return new BayesianProbabilityModel(probabilityMatrix,
        classPriorProbability);
  }

  private void observe(DoubleVector document, DoubleVector outcome,
      int numDistinctClasses, int[] tokenPerClass, int[] numDocumentsPerClass) {
    int predictedClass = outcome.maxIndex();
    if (numDistinctClasses == 2) {
      predictedClass = (int) outcome.get(0);
    }

    synchronized (probabilityMatrix) {
      tokenPerClass[predictedClass] += document.getLength();
      numDocumentsPerClass[predictedClass]++;
    }

    Iterator<DoubleVectorElement> iterateNonZero = document.iterateNonZero();
    while (iterateNonZero.hasNext()) {
      DoubleVectorElement next = iterateNonZero.next();
      // TODO this is a very granular lock that is acquired very often:
      // can this high contention be improved, e.g. by writing a temporary
      // vector and then just merging updates?
      synchronized (probabilityMatrix) {
        double currentCount = probabilityMatrix.get(predictedClass,
            next.getIndex());
        probabilityMatrix.set(predictedClass, next.getIndex(), currentCount
            + next.getValue());
      }
    }
  }

}
