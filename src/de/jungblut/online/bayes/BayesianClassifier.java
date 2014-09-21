package de.jungblut.online.bayes;

import java.util.Iterator;

import org.apache.commons.math3.util.FastMath;

import com.google.common.base.Preconditions;

import de.jungblut.classification.AbstractPredictor;
import de.jungblut.math.DoubleVector;
import de.jungblut.math.DoubleVector.DoubleVectorElement;
import de.jungblut.math.dense.DenseDoubleVector;

public class BayesianClassifier extends AbstractPredictor {

  private static final double LOW_PROBABILITY = FastMath.log(1e-8);

  private final BayesianProbabilityModel model;

  public BayesianClassifier(BayesianProbabilityModel model) {
    super();
    this.model = Preconditions.checkNotNull(model, "model");
  }

  @Override
  public DoubleVector predict(DoubleVector features) {
    return getProbabilityDistribution(features);
  }

  private double getProbabilityForClass(DoubleVector document, int classIndex) {
    double probabilitySum = 0.0d;
    Iterator<DoubleVectorElement> iterateNonZero = document.iterateNonZero();
    while (iterateNonZero.hasNext()) {
      DoubleVectorElement next = iterateNonZero.next();
      double wordCount = next.getValue();
      double probabilityOfToken = model.getProbabilityMatrix().get(classIndex,
          next.getIndex());
      if (probabilityOfToken == 0d) {
        probabilityOfToken = LOW_PROBABILITY;
      }
      probabilitySum += (wordCount * probabilityOfToken);
    }
    return probabilitySum;
  }

  private DenseDoubleVector getProbabilityDistribution(DoubleVector document) {

    int numClasses = model.getClassPriorProbability().getLength();
    DenseDoubleVector distribution = new DenseDoubleVector(numClasses);
    // loop through all classes and get the max probable one
    for (int i = 0; i < numClasses; i++) {
      double probability = getProbabilityForClass(document, i);
      distribution.set(i, probability);
    }

    double maxProbability = distribution.max();
    double probabilitySum = 0.0d;
    // we normalize it back
    for (int i = 0; i < numClasses; i++) {
      double probability = distribution.get(i);
      double normalizedProbability = FastMath.exp(probability - maxProbability
          + model.getClassPriorProbability().get(i));
      distribution.set(i, normalizedProbability);
      probabilitySum += normalizedProbability;
    }

    // since the sum is sometimes not 1, we need to divide by the sum
    distribution = (DenseDoubleVector) distribution.divide(probabilitySum);

    return distribution;
  }

}
