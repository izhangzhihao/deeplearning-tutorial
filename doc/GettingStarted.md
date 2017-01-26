Deep learning is a great tool that helps us efficiently summarize inherent patterns from tons of input data. I'd like to introduce DeepLearning.scala by letting the framework learn the common difference from Arithmetic progression.


##Background


**Input**:
 Arithmetic progression(AP) as:
``` val input: INDArray = Array(Array(0, 1, 2), Array(3, 6, 9), Array(13, 15, 17)).toNDArray``` 


**Output**: 
 Common Difference of the certain AP as: 
```val expectedOutput: INDArray = Array(Array(1), Array(3), Array(2)).toNDArray```

So here we want DeepLearning.scala to learn the common difference from the AP, i.e. ```{1} from {0, 1, 2} ``` 
in which `2-1 = 1-0 = 1 `


##Network Design


### Step 1: Install DeepLearning.scala

DeepLearning.scala is hosted on Maven Central repository.
If you use [sbt](http://www.scala-sbt.org), please add the following settings in your `build.sbt`:

``` sbt
libraryDependencies += "com.thoughtworks.deeplearning" %% "differentiableany" % "latest.release"

libraryDependencies += "com.thoughtworks.deeplearning" %% "differentiablenothing" % "latest.release"

libraryDependencies += "com.thoughtworks.deeplearning" %% "differentiableseq" % "latest.release"

libraryDependencies += "com.thoughtworks.deeplearning" %% "differentiabledouble" % "latest.release"

libraryDependencies += "com.thoughtworks.deeplearning" %% "differentiablefloat" % "latest.release"

libraryDependencies += "com.thoughtworks.deeplearning" %% "differentiablehlist" % "latest.release"

libraryDependencies += "com.thoughtworks.deeplearning" %% "differentiablecoproduct" % "latest.release"

libraryDependencies += "com.thoughtworks.deeplearning" %% "differentiableindarray" % "latest.release"

addCompilerPlugin("com.thoughtworks.implicit-dependent-type" %% "implicit-dependent-type" % "latest.release")

addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full)

fork := true
```

Note that this example does not run on Scala 2.12 because [nd4j](http://nd4j.org/) does not support Scala 2.12. Make sure there is not a setting like `scalaVersion := "2.12.1"` in your `build.sbt`.

See [Scaladex](https://index.scala-lang.org/thoughtworksinc/deeplearning.scala) for settings of other build tools.

### Step 2: Design your own neural network

DeepLearning.scala is also a language that we can use to create complex neural networks.

In the following sections, you will learn:
 * how to define types for a neural network
 * how to use a neural network as a predictor
 * how to create a neural network
 * how to train a neural network

#### 2.1 Define Input/Output Type 
Like a `scala.Function`, a neural network has its own input types and output types.

For example, the type of the neural network that accepts an N-dimensional array and returns another N-dimensional array is `(INDArray <=> INDArray)##T`.

``` scala
val myNeuralNetwork: (INDArray <=> INDArray)##T = ???
```

In `A <=> B`, A is the input type, and B is the output type. For the example above, both the input type and the output type are `INDArray`.

`##T` is a syntactic sugar to create implicit dependent types. See [implicit-dependent-type](https://github.com/ThoughtWorksInc/implicit-dependent-type) for more information about `##`.

In later sections of this article, you will replace `???` to a valid neural network.

#### 2.2 Use a neural network as a predictor

Like a normal `scala.Function`, if you pass the input data to the neural network, it will return some results.
You can use the `predict` method to invoke a neural network.

``` scala
val input: INDArray = Array(Array(0, 1, 2), Array(3, 6, 9), Array(13, 15, 17)).toNDArray
val predictionResult: INDArray = myNeuralNetwork.predict(input)
```

#### 2.3 Create a neural network

Same as the definition of a normal Scala function, the definition of neural network consists of a type definition for its parameter, a type definition for its return value, and a body that contains mathematical formulas, function-calls, and control flows.

``` scala
def createMyNeuralNetwork(implicit input: From[INDArray]##T): To[INDArray]##T = {
  val initialValueOfWeight = Nd4j.randn(3, 1)
  val weight: To[INDArray]##T = initialValueOfWeight.toWeight
  input dot weight
}
```

##### 2.3.1  Weight Intialization 

A neural network is trainable.
It means that some variables in the neural network can be changed automatically according to some goals. Those variables are called `weight`.
You can create weight variables via `toWeight` method, given its initial value.

In order to create a weight, you must create an `Optimizer`, which contains the rule that manages how the weight changes. See [Scaladoc](https://javadoc.io/page/com.thoughtworks.deeplearning/unidoc_2.11/latest/com/thoughtworks/deeplearning/DifferentiableINDArray$$Optimizers$.html) for a list of built-in optimizers.

``` scala
implicit def optimizer: Optimizer = new LearningRate {
  def currentLearningRate() = 0.001
}
```

##### 2.3.2 `From` and `To` placeholders

When you create a neural network, you have not actually evaluated it yet.
In fact, you only build its structure.
Variables in the neural network are placeholders,
which will be replaced with actual values in the future training or prediction process.

`From` is the placeholder type for input parameter,
and `To` is the placeholder type for return values and other local variables.

`From` must be `implicit` so that it is automatically generated when you create the neural network. Otherwise, you have to manually pass the `From` placeholder to `createMyNeuralNetwork`.

``` scala
val myNeuralNetwork: (INDArray <=> INDArray)##T = createMyNeuralNetwork
```

### Step 3: Train your Neuro Network

You have learned that weight will be automatically changed due to some goals.

In DeepLearning.scala, when we train a neural network, our goal should always be minimizing the absolute of the return value.

For example, if someone repeatedly call `myNeuralNetwork.train(Array(Array(0, 1, 2), Array(3, 6, 9), Array(13, 15, 17)).toNDArray)`,
the neural network would try to minimize `input dot weight`.
Soon `weight` would become an array of zeros in order to make `input dot weight` zeros,
and `myNeuralNetwork.predict(Array(Array(0, 1, 2), Array(3, 6, 9), Array(13, 15, 17)).toNDArray)` would return `Array(Array(0), Array(0), Array(0)).toNDArray`.

What if you expect `myNeuralNetwork.predict(Array(Array(0, 1, 2), Array(3, 6, 9), Array(13, 15, 17)).toNDArray)` to return `Array(Array(1), Array(3), Array(2)).toNDArray`?

You can create another neural network that evaluates how far between the result of `myNeuralNetwork` and your expectation. The new neural network is usually called **loss function**.

``` scala
def lossFunction(implicit pair: From[INDArray :: INDArray :: HNil]##T): To[Double]##T = {
  val input = pair.head
  val expectedOutput = pair.tail.head
  abs(myNeuralNetwork.compose(input) - expectedOutput).sum
}
```

When the `lossFunction` get trained continuously, its return value will be close to zero, and the result of  `myNeuralNetwork` must be close to the expected result at the same time.

Note the `lossFunction` accepts a placehold of `INDArray :: INDArray :: HNil` as its parameter, which is  a [shapeless](https://github.com/milessabin/shapeless)'s `HList` type.
The `HList` consists of two N-dimensional arrays.
The first array is the input data used to train the neural network, and the second array is the expected output.


``` scala
val input: INDArray = Array(Array(0, 1, 2), Array(3, 6, 9), Array(13, 15, 17)).toNDArray
val expectedOutput: INDArray = Array(Array(1), Array(3), Array(2)).toNDArray

for (iteration <- 0 until 2000) {
  lossFunction.train(input :: expectedOutput :: HNil)
}

// The loss should be close to zero
println(s"loss: ${ lossFunction.predict(input :: expectedOutput :: HNil) }")

// The prediction result should be close to expectedOutput
println(s"result: ${ myNeuralNetwork.predict(input) }")
```

## Summary

In this article, you have learned:
* to create neural networks dealing with complex data structures like `Double`, `INDArray` and `HList` like ordinary programming language
* to compose a neural network into a larger neural network
* to train a neural network
* to use a neural network as a predictor

The complete source codes used in this article are :

``` scala
import com.thoughtworks.deeplearning.DifferentiableHList._
import com.thoughtworks.deeplearning.DifferentiableDouble._
import com.thoughtworks.deeplearning.DifferentiableINDArray._
import com.thoughtworks.deeplearning.DifferentiableAny._
import com.thoughtworks.deeplearning.DifferentiableINDArray.Optimizers._
import com.thoughtworks.deeplearning.Lift._
import com.thoughtworks.deeplearning.Poly.MathFunctions._
import com.thoughtworks.deeplearning.Poly.MathOps
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import shapeless._

object GettingStarted extends App {

  implicit def optimizer: Optimizer = new LearningRate {
    def currentLearningRate() = 0.001
  }

  def createMyNeuralNetwork(implicit input: From[INDArray]##T): To[INDArray]##T = {
    val initialValueOfWeight = Nd4j.randn(3, 1)
    val weight: To[INDArray]##T = initialValueOfWeight.toWeight
    input dot weight
  }

  val myNeuralNetwork: (INDArray <=> INDArray)##T = createMyNeuralNetwork

  def lossFunction(implicit pair: From[INDArray :: INDArray :: HNil]##T): To[Double]##T = {
    val input = pair.head
    val expectedOutput = pair.tail.head
    abs(myNeuralNetwork.compose(input) - expectedOutput).sum
  }

  val input: INDArray = Array(Array(0, 1, 2), Array(3, 6, 9), Array(13, 15, 17)).toNDArray
  val expectedOutput: INDArray = Array(Array(1), Array(3), Array(2)).toNDArray

  for (iteration <- 0 until 2000) {
    lossFunction.train(input :: expectedOutput :: HNil)
  }

  // The loss should close to zero
  println(s"loss: ${ lossFunction.predict(input :: expectedOutput :: HNil) }")

  // The prediction result should close to Array(Array(1), Array(0)).toNDArray
  println(s"result: ${ myNeuralNetwork.predict(input) }")
}
```
