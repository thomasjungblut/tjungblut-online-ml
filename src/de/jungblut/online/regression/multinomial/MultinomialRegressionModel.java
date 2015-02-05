package de.jungblut.online.regression.multinomial;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import com.google.common.base.Preconditions;

import de.jungblut.online.ml.Model;
import de.jungblut.online.regression.RegressionModel;

public class MultinomialRegressionModel implements Model {

  private RegressionModel[] trainedModels;

  // deserialization constructor
  public MultinomialRegressionModel() {
  }

  public MultinomialRegressionModel(RegressionModel[] trainedModels) {
    this.trainedModels = Preconditions.checkNotNull(trainedModels, "weights");
    for (int i = 0; i < trainedModels.length; i++) {
      Preconditions.checkNotNull(trainedModels[i], "model at index " + i);
    }
  }

  @Override
  public void serialize(DataOutput out) throws IOException {
    out.writeInt(trainedModels.length);
    for (RegressionModel model : trainedModels) {
      model.serialize(out);
    }
  }

  @Override
  public MultinomialRegressionModel deserialize(DataInput in)
      throws IOException {
    trainedModels = new RegressionModel[in.readInt()];
    for (int i = 0; i < trainedModels.length; i++) {
      trainedModels[i] = new RegressionModel().deserialize(in);
    }

    return this;
  }

  public RegressionModel[] getModels() {
    return trainedModels;
  }

}
