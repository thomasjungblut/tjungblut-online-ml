tjungblut-online-ml
===================

This is my online machine learning library. Over the next few months I will put some algorithms from my main common library into a streaming fashion and move them into this repository.

Everything will be built upon the Java 8 streams and the algorithms are specifically designed to make use of the streams feature, also I aim scale this library vertically by using parallel streams wherever possible.

This is not a distributed system, but its parts can be reused with any MapReduce or BSP implementation (e.g. Hadoop MR/Hama BSP), making it horizontally scalable for terabyte/petabyte datasets as well.

For the future, I also want to support infinite streams where the training can be done asynchronously with using the model.

Supported Algorithms
===================

- [x] Multinomial Naive Bayes
 - [ ] Complement Naive Bayes
- [x] Stochastic Gradient Descent 
 - [x] logistic regression
 - [x] linear regression (least squares)
 - [ ] MaxEnt (might be part of the Multilayer Perceptron below)
 - [ ] MaxEnt Markov Models
 - [ ] Lasso (l1 norm)
 - [x] Ridge Regression (l2 norm)
 - [ ] SVM (Hinge Loss)
 - [ ] CG Support
 - [ ] Adagrad
- [ ] Multilayer Perceptron
- [ ] RBM
- [ ] KNN
- [ ] Canopy Clustering

Sample Usage
===================

Stochastic Gradient Descent (logistic regression)
-------------------------------------------------

A very simplistic example is the SGD (stochastic gradient descent) classifier, the following example will train a logistic regression model:

```java
// use a gradient descent with a learning rate of 0.1
StochasticGradientDescent min = StochasticGradientDescentBuilder.create(0.1).build();

// generate the data, note that the features must include a bias (constant 1) if you want to have one
List<FeatureOutcomePair> data = generateData();

RegressionLearner learner = new RegressionLearner(min, new SigmoidActivationFunction(), new LogisticErrorFunction());
// do 5 passes over all data in a stream, the default is 1
learner.setNumPasses(5);

// train the model by supplying the stream
RegressionModel model = learner.train(() -> data.stream());

// print the weights
System.out.println(model.getWeights());
```

The StochasticGradientDescent class makes use of parallel streams, so if you want to change the size of the internal fork-join pool that it uses, you can set:

> -Djava.util.concurrent.ForkJoinPool.common.parallelism=30

to whatever maximizes your throughput in updates per s.


Do Predictions
--------------

You can also do predictions with the model using a Classifier:

```java
RegressionClassifier classifier = new RegressionClassifier(model);
// add the bias to the feature and predict it
DoubleVector prediction = classifier.predict(new DenseDoubleVector(new double[] { 1, 25d, 25d }))
// print the prediction
System.out.println(prediction);
```

Serialize your model
--------------------

If you want to save the model to a file, you can use the serialization API offered by the model:

```java
try (DataOutputStream dos = new DataOutputStream(new FileOutputStream("/tmp/model.bin"))){
	model.serialize(dos);
}
```

Deserialization works in the same way:
```java
RegressionModel model = new RegressionModel();
try (DataInputStream dis = new DataInputStream(new FileInputStream("/tmp/model.bin"))){
	model.deserialize(dis);
}
// take dis
```


License
-------

Since I am Apache committer, I consider everything inside of this repository 
licensed by Apache 2.0 license, although I haven't put the usual header into the source files.

If something is not licensed via Apache 2.0, there is a reference or an additional licence header included in the specific source file.


Build
-----

You will need Java 8 to build this library.

You can simply build with:
 
> mvn clean package install

The created jars contains debuggable code + sources + javadocs.

If you want to skip testcases you can use:

> mvn clean package install -DskipTests

If you want to skip the signing process you can do:

> mvn clean package install -Dgpg.skip=true
