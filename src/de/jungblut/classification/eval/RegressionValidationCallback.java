package de.jungblut.classification.eval;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import de.jungblut.classification.eval.Evaluator.EvaluationResult;
import de.jungblut.math.DoubleVector;
import de.jungblut.online.minimizer.PassFinishedCallback;
import de.jungblut.online.minimizer.ValidationFinishedCallback;
import de.jungblut.online.ml.FeatureOutcomePair;
import de.jungblut.online.regression.RegressionClassifier;
import de.jungblut.online.regression.RegressionLearner;
import de.jungblut.online.regression.RegressionModel;

public class RegressionValidationCallback implements
    ValidationFinishedCallback, PassFinishedCallback {

  private static final Logger LOG = LogManager
      .getLogger(RegressionValidationCallback.class);

  private EvaluationResult currentResult;
  private RegressionLearner learner;

  public RegressionValidationCallback(RegressionLearner learner) {
    this.learner = learner;
    setupNewEvaluationResult();
  }

  @Override
  public boolean onPassFinished(int pass, long iteration, double cost,
      DoubleVector currentWeights) {

    LOG.info("Evaluation | Pass " + pass + " | Iteration " + iteration);

    currentResult.print(LOG);

    setupNewEvaluationResult();

    return true;
  }

  @Override
  public void onValidationFinished(int pass, long iteration, double cost,
      DoubleVector currentWeights, FeatureOutcomePair pair) {

    RegressionModel model = learner.createModel(currentWeights);
    RegressionClassifier classifier = new RegressionClassifier(model);
    currentResult.testSize++;
    DoubleVector predict = classifier.predict(pair.getFeature());
    Evaluator.observeBinaryClassificationElement(classifier, null,
        currentResult, pair.getOutcome(), predict);

  }

  private void setupNewEvaluationResult() {
    currentResult = new EvaluationResult();
    currentResult.numLabels = 2;
  }
}
