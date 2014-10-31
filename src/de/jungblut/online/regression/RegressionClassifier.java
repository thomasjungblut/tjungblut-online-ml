package de.jungblut.online.regression;

import com.google.common.base.Preconditions;

import de.jungblut.classification.AbstractPredictor;
import de.jungblut.math.DoubleVector;
import de.jungblut.math.activation.ActivationFunction;
import de.jungblut.math.dense.SingleEntryDoubleVector;

/**
 * Classifier for regression model. Takes a model or the atomic parts of it and
 * predicts the outcome for a given feature.
 * 
 * @author thomas.jungblut
 *
 */
public class RegressionClassifier extends AbstractPredictor {

  private final RegressionModel model;

  public RegressionClassifier(RegressionModel model) {
    this.model = Preconditions.checkNotNull(model, "model");
  }

  public RegressionClassifier(DoubleVector weights, ActivationFunction function) {
    this(new RegressionModel(weights, function));
  }

  @Override
  public DoubleVector predict(DoubleVector feature) {
    Preconditions.checkArgument(feature.getDimension() == model.getWeights()
        .getDimension(),
        "feature dimension must match model weight dimension! Feature: "
            + feature.getDimension() + " != Model: "
            + model.getWeights().getDimension());

    double result = model.getActivationFunction().apply(
        feature.dot(model.getWeights()));

    return new SingleEntryDoubleVector(result);
  }

}
