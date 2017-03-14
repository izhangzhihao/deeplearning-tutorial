package com.thoughtworks.deeplearning.tutorial

import com.thoughtworks.deeplearning
import org.nd4j.linalg.api.ndarray.INDArray
import com.thoughtworks.deeplearning.DifferentiableHList._
import com.thoughtworks.deeplearning.DifferentiableDouble._
import com.thoughtworks.deeplearning.DifferentiableINDArray._
import com.thoughtworks.deeplearning.DifferentiableAny._
import com.thoughtworks.deeplearning.DifferentiableInt._
import com.thoughtworks.deeplearning.DifferentiableSeq._
import com.thoughtworks.deeplearning.DifferentiableINDArray.Optimizers._
import com.thoughtworks.deeplearning.Layer.Batch.Aux
import com.thoughtworks.deeplearning._
import com.thoughtworks.deeplearning.Layer.{Aux, Batch}
import com.thoughtworks.deeplearning.Poly.MathFunctions.{exp, log, max}
import com.thoughtworks.deeplearning.Symbolic.Layers.Identity
import com.thoughtworks.deeplearning.Symbolic._
import com.thoughtworks.deeplearning.Poly.MathFunctions._
import com.thoughtworks.deeplearning.Poly.MathMethods.{*, /}
import com.thoughtworks.deeplearning.Poly.MathOps
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.cpu.nativecpu.NDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.factory.Nd4j.PadMode
import org.nd4j.linalg.factory.Nd4j.PadMode.EDGE
import org.nd4j.linalg.indexing.{INDArrayIndex, NDArrayIndex}
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4s.Implicits._
import plotly.Scatter
import shapeless._
import plotly.Plotly._
import plotly._
import shapeless.OpticDefns.compose

import scala.annotation.tailrec
import scala.collection.immutable.IndexedSeq
import Utils._
import com.thoughtworks.deeplearning.DifferentiableINDArray.Layers.Weight
import org.joda.time.LocalTime

/**
  * Created by 张志豪 on 2017/2/6.
  */
object CNNs extends App {

  var isAEpochDone = false

  implicit val optimizerFactory = new DifferentiableINDArray.OptimizerFactory {
    override def ndArrayOptimizer(weight: Weight): Optimizer = {

      new LearningRate with L2Regularization with Adam { //NesterovMomentum

        var learningRate = 0.00003

        override protected def l2Regularization: Double = 0.00003

        override protected def currentLearningRate(): Double = {
          learningRate = if (isAEpochDone) {
            isAEpochDone = false
            println("setting isUpdateLearningRate to : " + isAEpochDone)
            println("before update learningRate : " + learningRate)
            learningRate * 0.75
          } else {
            learningRate
          }
          learningRate
        }
      }
    }
  }

  //CIFAR10中的图片共有10个分类(airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck)
  val NumberOfClasses: Int = 10

  val NumberOfTestSize = 100

  //加载测试数据，我们读取100条作为测试数据
  val testNDArray =
    ReadCIFAR10ToNDArray.readFromResource(
      "/cifar-10-batches-bin/test_batch.bin",
      NumberOfTestSize)

  val testData = testNDArray.head

  val testExpectLabel = testNDArray.tail.head

  val testExpectLabelVectorized =
    Utils.makeVectorized(testExpectLabel, NumberOfClasses)

  val MiniBatchSize = 64

  val Depth = List.fill(17)(3)

  val InputSize = 32 // W 输入数据尺寸

  val KernelNumber = List.fill(17)(3) //卷积核的数量

  val Stride = 1 // 步长

  val Padding = 1 //零填充数量

  val KernelSize = 3 //F 卷积核的空间尺寸

  val reshapedTestData =
    testData.reshape(NumberOfTestSize, 3, InputSize, InputSize)

  def convolutionThenRelu(numberOfInputKernels: Int,
                          numberOfOutputKernels: Int)(
      implicit input: INDArray @Symbolic): INDArray @Symbolic = {
    val weight: INDArray @Symbolic =
      (Nd4j.randn(
        Array(numberOfOutputKernels,
              numberOfInputKernels,
              KernelSize,
              KernelSize)) *
        math.sqrt(2.0 / numberOfInputKernels / KernelSize / KernelSize)).toWeight
    //When using RELUs, make sure biases are initialised with small *positive* values for example 0.1
    val bias = Nd4j.ones(numberOfOutputKernels).toWeight * 0.1

    val convResult =
      conv2d(input, weight, bias, (3, 3), (1, 1), (1, 1))
    max(convResult, 0.0)
  }

  //  def maxPool(implicit input: INDArray @Symbolic): INDArray @Symbolic = {
  //    input
  //      .im2col(Array(2, 2), Array(2, 2), Array(0, 0))
  //      .permute(0, 1, 4, 5, 2, 3)
  //      .maxPool(4, 5)
  //  }

  def softmax(implicit scores: INDArray @Symbolic): INDArray @Symbolic = {
    val expScores = exp(scores)
    expScores / expScores.sum(1)
  }

  def fullyConnectedThenSoftmax(inputSize: Int, outputSize: Int)(
      implicit input: INDArray @Symbolic): INDArray @Symbolic = {
    val imageCount = input.shape(0)

    val weight =
      (Nd4j.randn(inputSize, outputSize) / math.sqrt(outputSize)).toWeight
    val bias = Nd4j.zeros(outputSize).toWeight

    softmax.compose(
      (input.reshape(imageCount, inputSize.toLayer) dot weight) + bias)
  }

  def hiddenLayer(implicit input: INDArray @Symbolic): INDArray @Symbolic = {
    @tailrec
    def convFunction(timesToRun: Int,
                     timesNow: Int,
                     input2: INDArray @Symbolic): INDArray @Symbolic = {
      if (timesToRun <= 0) {
        input2
      } else {
        convFunction(
          timesToRun - 1,
          timesNow + 1,
          convolutionThenRelu(Depth(timesNow * 2 + 1),
                              KernelNumber(timesNow * 2 + 1)).compose(
            convolutionThenRelu(Depth(timesNow * 2),
                                KernelNumber(timesNow * 2)).compose(input2)
          )
        )
      }
    }

    val recLayer = convFunction(8, 0, input)

    fullyConnectedThenSoftmax(32 * 32 * 3, 10).compose(recLayer)
  }

  val predictor = hiddenLayer

  def crossEntropyLossFunction(
      implicit pair: (INDArray :: INDArray :: HNil) @Symbolic)
    : Double @Symbolic = {
    val score = pair.head
    val label = pair.tail.head
    -(label * log(score * 0.9 + 0.1) + (1.0 - label) * log(1.0 - score * 0.9)).mean
  }

  def network(implicit pair: (INDArray :: INDArray :: HNil) @Symbolic)
    : Double @Symbolic = {
    val input = pair.head
    val label = pair.tail.head
    val score: INDArray @Symbolic = predictor.compose(input)
    crossEntropyLossFunction.compose(score :: label :: HNil.toLayer)
  }

  val trainNetwork = network

  val random = new util.Random

  def trainData(randomIndexArray: Array[Int],
                isComputeAccuracy: Boolean): (Double, Double, Double) = {
    val trainNDArray :: expectLabel :: shapeless.HNil =
      ReadCIFAR10ToNDArray.getSGDTrainNDArray(randomIndexArray)
    val input =
      trainNDArray.reshape(MiniBatchSize, 3, InputSize, InputSize)

    val expectLabelVectorized =
      Utils.makeVectorized(expectLabel, NumberOfClasses)
    val trainLoss = trainNetwork.train(input :: expectLabelVectorized :: HNil)

    if (isComputeAccuracy) {
      val trainResult: INDArray = predictor.predict(input)

      val trainAccuracy = Utils.getAccuracy(trainResult, expectLabel) / 100

      val testResult: INDArray = predictor.predict(reshapedTestData)

      val testAccuracy = Utils.getAccuracy(testResult, testExpectLabel) / 100
      println(
        s"train accuracy : $trainAccuracy ,\t\ttest accuracy : $testAccuracy ,\t\ttrain loss : $trainLoss ")
      (trainLoss, trainAccuracy, testAccuracy)
    } else {
      println(s"train loss : $trainLoss ")
      (trainLoss, 0, 0)
    }
  }

  val startTime = LocalTime.now()

  val resultTuple: Seq[(Double, Double, Double)] =
    (
      for (blocks <- 0 to 50) yield {
        if (blocks % 5 == 0 && blocks != 0) {
          //一个epoch
          isAEpochDone = true
        }
        val randomIndex = random
          .shuffle[Int, IndexedSeq](0 until 10000) //https://issues.scala-lang.org/browse/SI-6948
          .toArray
        for (times <- 0 until 10000 / MiniBatchSize) yield {
          val randomIndexArray =
            randomIndex.slice(times * MiniBatchSize,
                              (times + 1) * MiniBatchSize)
          trainData(randomIndexArray, isAEpochDone)
        }
      }
    ).flatten

  val endTime = LocalTime.now()

  println(
    "start at :" + startTime.toString() + "; end at :" + endTime.toString())

//  val resultTuple: Seq[(Double, Double, Double)] =
//    for (indexer <- 0 until 1000) yield {
//      if (indexer % 100 == 0 && indexer != 0) {
//        isUpdateLearningRate = true
//        println("setting isUpdateLearningRate to : " + isUpdateLearningRate)
//      }
//      trainData(Array(1))
//    }

  val (trainLossSeq, trainAccuracySeq, testAccuracySeq) =
    resultTuple.unzip3

  val filteredTrainAccuracySeq = 0.0 +: trainAccuracySeq.filter(item =>
    item != 0)

  val filteredTestAccuracySeq = 0.0 +: testAccuracySeq.filter(item =>
    item != 0)

  val interval = 5 * 10000 / MiniBatchSize

  val plot = Seq(
    Scatter(trainLossSeq.indices, trainLossSeq, name = "train loss"),
    Scatter(trainLossSeq.indices by interval,
            filteredTrainAccuracySeq,
            name = "train accuracy"),
    Scatter(trainLossSeq.indices by interval,
            filteredTestAccuracySeq,
            name = "test accuracy")
  )

  plot.plot(title = "train loss,train accuracy,test accuracy by time")
}
