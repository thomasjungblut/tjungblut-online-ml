package de.jungblut.online.regression.multinomial;

import com.google.common.base.Preconditions;

import de.jungblut.classification.AbstractPredictor;
import de.jungblut.math.DoubleVector;
import de.jungblut.math.dense.DenseDoubleVector;
import de.jungblut.online.regression.RegressionClassifier;

/**
 * Classifier for multinomial regression.
 * 
 * @author thomas.jungblut
 *
 */
public class MultinomialRegressionClassifier extends AbstractPredictor {

  private final RegressionClassifier[] classifier;
  private boolean normalize;

  /**
   * Constructs a new multinomial regression classifier that does normalization
   * over independent predictions.
   * 
   * @param model the trained model.
   */
  public MultinomialRegressionClassifier(MultinomialRegressionModel model) {
    this(model, true);
  }

  /**
   * Constructs a new multinomial regression classifier that does normalization
   * over independent predictions by summing over the predictions and dividing
   * each entry.
   * 
   * @param model the trained model.
   * @param normalize true for normalizing the output.
   * 
   */
  public MultinomialRegressionClassifier(MultinomialRegressionModel model,
      boolean normalize) {
    this.normalize = normalize;
    Preconditions.checkNotNull(model, "model");
    this.classifier = new RegressionClassifier[model.getModels().length];
    for (int i = 0; i < model.getModels().length; i++) {
      classifier[i] = new RegressionClassifier(model.getModels()[i]);
    }
  }

  @Override
  public DoubleVector predict(DoubleVector feature) {

    DoubleVector mesh = new DenseDoubleVector(classifier.length);
    for (int i = 0; i < classifier.length; i++) {
      RegressionClassifier clf = classifier[i];
      DoubleVector prediction = clf.predict(feature);
      Preconditions.checkArgument(prediction.getDimension() == 1,
          "Prediction only works for a single dimensional output! Given "
              + prediction.getDimension());
      mesh.set(i, prediction.get(0));
    }

    if (normalize) {
      double sum = mesh.sum();
      if (sum != 0d) {
        mesh = mesh.divide(sum);
      }
    }

    return mesh;
  }

}
