tjungblut-online-ml
===================

This is my online machine learning library. Over the next few months I will put some algorithms from my main common library into a streaming fashion and move them into this repository.

Everything will be built upon the Java 8 streams and the algorithms are specifically designed to make use of the streams feature, also I aim scale this library vertically by using parallel streams wherever possible.

This is not a distributed system, but its parts can be reused with any MapReduce or BSP implementation (e.g. Hadoop MR/Hama BSP), making it horizontally scalable for terabyte/petabyte datasets as well.

For the future, I also want to support infinite streams where the training can be done asynchronously with using the model.

Supported Algorithms
===================

- [x] Multinomial Naive Bayes
- [x] Stochastic Gradient Descent
 - [x] Logistic regression
 - [x] Linear regression (least squares)
 - [x] Multinomial regression (one vs. all)
 - [x] Maximum Margin (hinge loss)
 - [x] Lasso (l1 norm)
 - [x] Ridge Regression (l2 norm)
 - [x] FTRL-Proximal
 - [x] Adam
 - [ ] CG
 - [ ] Sample-based Adaptive Learning Rates
 - [ ] Shuffled input streams
 - [ ] Multilayer Perceptron
- [ ] Graphite Bindings

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

RegressionLearner learner = new RegressionLearner(min, new SigmoidActivationFunction(), new LogLoss());
// do 5 passes over all data in a stream, the default is 1
learner.setNumPasses(5);

// train the model by supplying the stream
RegressionModel model = learner.train(() -> data.stream());

// print the weights
System.out.println(model.getWeights());
```

The StochasticGradientDescent class makes use of parallel streams, so if you want to change the size of the internal fork-join pool that it uses, you can set:

> -Djava.util.concurrent.ForkJoinPool.common.parallelism=30

to whatever maximizes your throughput in updates per second.


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


Avazu Click-Through Rate Prediction
-------------------------------------

A prime example to use this streaming library for is to do CTR predictions. Below code takes the data from the [Avazu CTR prediction challenge on kaggle](https://www.kaggle.com/c/avazu-ctr-prediction/).
It is using simple feature hashing and FTRL logistic regression.

```java

public class AvazuCtrPrediction {

  private static final int NUM_COLUMNS = 24;
  private static final int SPARSE_HASH_DIMENSION = 2 << 24;
  private static final Pattern SPLITTER = Pattern.compile(",");
  private static final int BUFFER_SIZE = 1024 * 1024 * 5;
  private static final String TRAINING_SET_PATH = "/home/user/datasets/ctr/train.gz";

  private static final SingleEntryDoubleVector POSITIVE_CLASS //
  = new SingleEntryDoubleVector(1d);
  private static final SingleEntryDoubleVector NEGATIVE_CLASS //
  = new SingleEntryDoubleVector(0d);

  private static FeatureOutcomePair parseFeature(String line, String[] header) {

    final int shift = 2;
    String[] split = SPLITTER.split(line);
    Preconditions.checkArgument(split.length == NUM_COLUMNS,
        "line doesn't match expected size");

    // turn the date into the hour
    split[2] = split[2].substring(6);

    // prepare the tokens for feature hashing
    String[] tokens = new String[split.length - shift];
    for (int i = 0; i < tokens.length; i++) {
      tokens[i] = header[i + shift] + "_" + split[i + shift];
    }

    // hash them with 128 bit murmur3
    DoubleVector feature = VectorizerUtils.sparseHashVectorize(tokens, Hashing
        .murmur3_128(), () -> new SequentialSparseDoubleVector(
        SPARSE_HASH_DIMENSION));

    // fix the first element to be the bias
    feature.set(0, 1d);

    return new FeatureOutcomePair(feature,
        split[1].equals("0") ? NEGATIVE_CLASS : POSITIVE_CLASS);
  }

  private static Stream<FeatureOutcomePair> setupStream() {
    try {
      @SuppressWarnings("resource")
      BufferedReader reader = new BufferedReader(new InputStreamReader(
          new GZIPInputStream(new FileInputStream(TRAINING_SET_PATH),
              BUFFER_SIZE), Charset.defaultCharset()));

      // consume the header first
      final String[] header = SPLITTER.split(reader.readLine());
      // yield the stream for everything that comes after
      return reader.lines().map((s) -> parseFeature(s, header));
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  public static void main(String[] args) throws IOException {

    StochasticGradientDescent sgd = StochasticGradientDescentBuilder
        .create(0.01) // learning rate
        .holdoutValidationPercentage(0.05d) // 5% as validation set
        .historySize(10_000) // keep 10k samples to compute relative improvement
        .weightUpdater(new AdaptiveFTRLRegularizer(1, 1, 1)) // FTRL updater
        .progressReportInterval(1_000_000) // report every n iterations
        .build();

    // simple regression with Sigmoid and LogLoss
    RegressionLearner learner = new RegressionLearner(sgd,
        new SigmoidActivationFunction(), new LogLoss());

    // you are able to trade speed with memory usage!
    // using sparse weights should use roughly 400mb, vs. 3gb of dense weights.
    // however, dense weights are 10x faster in this case.
    // learner.useSparseWeights();

    // do two full passes over the data
    learner.setNumPasses(2);
    learner.verbose();

    Stopwatch sw = Stopwatch.createStarted();

    // train the model
    RegressionModel model = learner.train(() -> setupStream());

    // output the weights
    model.getWeights().iterateNonZero().forEachRemaining(System.out::println);

    System.out.println("Time taken: " + sw.toString());

  }

}

```

You should see similar output to the one below (verbosity omitted):

```
Pass 0 | Iteration 1000000 | Validation Cost: 0.392678 | Training Cost: 0.414122 | Avg Improvement: -5.06327e-07 | Iterations/s: 83333.3
---
Pass 0 | Iteration 38000000 | Validation Cost: 0.400473 | Training Cost: 0.422624 | Avg Improvement: 1.80911e-08 | Iterations/s: 93596.1
Pass Summary 0 | Iteration 38406639 | Validation Cost: 0.400674 | Training Cost: 0.422823 | Iterations/s: 93446.8  | Total Time Taken: 6.854 min
Pass 1 | Iteration 1000000 | Validation Cost: 0.377576 | Training Cost: 0.403000 | Avg Improvement: -4.86178e-07 | Iterations/s: 93602.5
---
Pass 1 | Iteration 38000000 | Validation Cost: 0.396356 | Training Cost: 0.418665 | Avg Improvement: 2.13002e-08 | Iterations/s: 93981.1
Pass Summary 1 | Iteration 38406798 | Validation Cost: 0.396473 | Training Cost: 0.418885 | Iterations/s: 94018.9  | Total Time Taken: 13.63 min

0 -> -0.2179566321842441
19 -> -0.008019816916453593
40 -> 0.011372162311553732
72 -> 0.009959662995812498
75 -> 0.01552960872382698
91 -> 0.01122045025790969
114 -> -0.058862175500113786
---
33554352 -> -0.013848560325345247
33554353 -> -7.612543688230045E-4
33554422 -> -0.003704587068022832
Time taken: 13.94 min
```



MNIST Multinomial Logistic Regression
-------------------------------------

A very simply code example for training the multinomial logistic regression is on the MNIST dataset.
Here we use the data from the [digit recognizer kaggle competetion](http://www.kaggle.com/c/digit-recognizer).

```java

    Dataset trainingSet = MNISTReader.readMNISTTrainImages("/home/user/datasets/mnist/kaggle/train.csv");

    IntFunction<RegressionLearner> factory = (i) -> {
    	  // take care of not sharing any state from the outside, since classes are trained in parallel
        StochasticGradientDescent minimizer = StochasticGradientDescentBuilder
        .create(0.01)
        .holdoutValidationPercentage(0.1d)
        .weightUpdater(new L2Regularizer(0.1))
        .progressReportInterval(1_000_000)
        .build();
      RegressionLearner learner = new RegressionLearner(minimizer,
          new SigmoidActivationFunction(), new LogLoss());
      learner.setNumPasses(50);
      learner.verbose();
      return learner;
    };

    MultinomialRegressionLearner learner = new MultinomialRegressionLearner(factory);
    learner.verbose();

    MultinomialRegressionModel model = learner.train(() -> trainingSet.asStream());
    MultinomialRegressionClassifier clf = new MultinomialRegressionClassifier(model);    
    // do some classifications

```

The accuracy and confusion matrix on a test set looks like this:

```
 31280 /  42000 acc: 0.7447619047619047
 3680    5  111  110   19  187  152   23   95   42 <-   744   17%	 0
    6 4553  212  254  127  157   52  187  544  137 <-  1676   27%	 1
   53   37 2916  344   53   46  182   33   54   17 <-   819   22%	 2
   31   11  328 3052   77  447   24  208  378   82 <-  1586   34%	 3
   65   15  121   66 3249  402   46  168   84  675 <-  1642   34%	 4
  130   28   67  195  104 1957  107    1  323   17 <-   972   33%	 5
   71   14  181   48   60   91 3511    3   29    2 <-   499   12%	 6
   27    7   74  119   22   86    0 3481   38  698 <-  1071   24%	 7
   59   11  143   88   57  271   45    4 2438   75 <-   753   24%	 8
   10    3   24   75  304  151   18  293   80 2443 <-   958   28%	 9
```

License
-------

Since I am Apache committer, I consider everything inside of this repository
licensed by Apache 2.0 license, although I haven't put the usual header into the source files.

If something is not licensed via Apache 2.0, there is a reference or an additional licence header included in the specific source file.

Maven
-----

If you use maven, you can get the latest release using the following dependency:

```
 <dependency>
     <groupId>de.jungblut.ml</groupId>
     <artifactId>tjungblut-online-ml</artifactId>
     <version>0.2</version>
 </dependency>
```

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
