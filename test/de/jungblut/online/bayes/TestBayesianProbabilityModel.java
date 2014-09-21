package de.jungblut.online.bayes;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;

import org.junit.Test;

public class TestBayesianProbabilityModel {

  @Test
  public void testSerDe() throws Exception {
    BayesianProbabilityModel model = TestNaiveBayesLearner.getTrainedModel();
    TestNaiveBayesLearner.checkModel(model);

    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    DataOutputStream dos = new DataOutputStream(baos);
    model.serialize(dos);

    ByteArrayInputStream bais = new ByteArrayInputStream(baos.toByteArray());
    DataInputStream dis = new DataInputStream(bais);
    model = new BayesianProbabilityModel();
    BayesianProbabilityModel deserialized = model.deserialize(dis);

    TestNaiveBayesLearner.checkModel(deserialized);
  }

}
