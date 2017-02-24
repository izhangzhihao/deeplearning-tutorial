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
import com.thoughtworks.deeplearning.Lift.Layers.Identity
import com.thoughtworks.deeplearning.Lift._
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
import plotly.Scatter
import shapeless._
import plotly.Plotly._
import plotly._
import shapeless.OpticDefns.compose
import scala.annotation.tailrec
import scala.collection.immutable.IndexedSeq

/**
  * Created by 张志豪 on 2017/2/6.
  */
object CNNs extends App {

  implicit val optimizerFactory = new DifferentiableINDArray.OptimizerFactory {
    override def ndArrayOptimizer(
        weight: DifferentiableINDArray.Layers.Weight): L2Regularization = {
      new DifferentiableINDArray.Optimizers.L2Regularization {
        override protected def l2Regularization = 0.03

        var learningRate = 0.0001

        override protected def currentLearningRate(): Double = {
          learningRate
        }

        override def updateNDArray(oldValue: INDArray,
                                   delta: INDArray): INDArray = {
          learningRate *= 0.9995
          super.updateNDArray(oldValue, delta)
        }
      }
    }
  }

  //CIFAR10中的图片共有10个分类(airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck)
  val NumberOfClasses: Int = 10

  //加载测试数据，我们读取100条作为测试数据
  val testNDArray =
    ReadCIFAR10ToNDArray.readFromResource(
      "/cifar-10-batches-bin/test_batch.bin",
      100)

  val test_data = testNDArray.head

  val test_expect_result = testNDArray.tail.head

  val test_p = Utils.makeVectorized(test_expect_result, NumberOfClasses)

  val MiniBatchSize = 64

  val Depth = Seq(3, 4, 8, 16, 32)

  val InputSize = 32 // W 输入数据尺寸

  val KernelNumber = Seq(4, 8, 16, 32) //卷积核的数量

  val Stride = 1 // 步长

  val Padding = 1 //零填充数量

  val KernelSize = 3 //F 卷积核的空间尺寸

  val reshapedTestData = test_data.reshape(100, 3, InputSize, InputSize)

  def convolutionThenRelu(depth: Int, kernelNumber: Int)(
      implicit input: From[INDArray]##T): To[INDArray]##T = {
    val imageCount = input.shape(0)
    val inputSize = input.shape(2)

    val outputSize = inputSize

    val weight =
      (Nd4j.randn(Array(kernelNumber, depth, KernelSize, KernelSize)) /
        math.sqrt(kernelNumber)).toWeight * 0.1

    val bias = Nd4j.zeros(kernelNumber).toWeight

    val colRow =
      input.im2col(Array(KernelSize, KernelSize),
                   Array(Stride, Stride),
                   Array(Padding, Padding))

    val permuteCol = colRow.permute(0, 2, 3, 1, 4, 5)

    val col2dRow = permuteCol.reshape(
      imageCount * outputSize * outputSize,
      (depth * KernelSize * KernelSize).toLayer)

    val reshapedW =
      weight.reshape(KernelSize * KernelSize * depth, kernelNumber)

    val res = max((col2dRow dot reshapedW) + bias, 0.0)

    val result =
      res.reshape(imageCount, outputSize, outputSize, kernelNumber.toLayer)

    result.permute(0, 3, 1, 2)
  }

  def maxPool(implicit input: From[INDArray]##T): To[INDArray]##T = {
    input
      .im2col(Array(2, 2), Array(2, 2), Array(0, 0))
      .permute(0, 1, 4, 5, 2, 3)
      .maxPool(4, 5)
  }

  def softmax(implicit scores: From[INDArray]##T): To[INDArray]##T = {
    val expScores = exp(scores)
    expScores / expScores.sum(1)
  }

  def fullyConnectedThenSoftmax(inputSize: Int, outputSize: Int)(
      implicit input: From[INDArray]##T): To[INDArray]##T = {
    val imageCount = input.shape(0)

    val weight =
      (Nd4j.randn(inputSize, outputSize) / math.sqrt(outputSize)).toWeight
    val bias = Nd4j.zeros(outputSize).toWeight

    softmax.compose(
      (input.reshape(imageCount, inputSize.toLayer) dot weight) + bias)
  }

  def hiddenLayer(implicit input: From[INDArray]##T): To[INDArray]##T = {

    @tailrec
    def convFunction(timesToRun: Int,
                     timesNow: Int,
                     input2: To[INDArray]##T): To[INDArray]##T = {
      if (timesToRun <= 0) {
        input2
      } else {
        convFunction(
          timesToRun - 1,
          timesNow + 1,
          maxPool.compose(
            convolutionThenRelu(Depth(timesNow * 2 + 1),
                                KernelNumber(timesNow * 2 + 1)).compose(
              convolutionThenRelu(Depth(timesNow * 2),
                                  KernelNumber(timesNow * 2)).compose(input2)
            )
          )
        )
      }
    }

    val recLayer = convFunction(2, 0, input)

    fullyConnectedThenSoftmax(32 * 8 * 8, 10).compose(recLayer)
  }

  val predictor = hiddenLayer

  def crossEntropyLossFunction(
      implicit pair: From[INDArray :: INDArray :: HNil]##T): To[Double]##T = {
    val score = pair.head
    val label = pair.tail.head
    -(label * log(score * 0.9 + 0.1) + (1.0 - label) * log(1.0 - score * 0.9)).sum
  }

  def network(
      implicit pair: From[INDArray :: INDArray :: HNil]##T): To[Double]##T = {
    val input = pair.head
    val label = pair.tail.head
    val score: To[INDArray]##T = predictor.compose(input)

    val hnilLayer: To[HNil]##T = HNil
    crossEntropyLossFunction.compose(score :: label :: hnilLayer)
  }

  val trainNetwork = network

  val random = new util.Random

  def trainData(randomIndexArray: Array[Int]): (Double, Double, Double) = {
    val trainNDArray :: expectLabel :: shapeless.HNil =
      ReadCIFAR10ToNDArray.getSGDTrainNDArray(randomIndexArray)
    val input =
      trainNDArray.reshape(MiniBatchSize, 3, InputSize, InputSize)
    val expectResult = Utils.makeVectorized(expectLabel, NumberOfClasses)

    val loss = trainNetwork.train(input :: expectResult :: HNil)
    println(s"loss : $loss")

    val trainResult: INDArray = predictor.predict(input)
    val trainAccuracy = Utils.getAccuracy(trainResult, expectLabel) * 100
    println(s"the train predict result is $trainAccuracy %")

    val testResult: INDArray = predictor.predict(reshapedTestData)
    val testAccuracy = Utils.getAccuracy(testResult, test_expect_result) * 100
    println(s"the test predict result is $testAccuracy %")

    (loss, trainAccuracy, testAccuracy)
  }

  val resultTuple: Seq[(Double, Double, Double)] =
    (for (_ <- 0 until 5) yield {
      val randomIndex = random
        .shuffle[Int, IndexedSeq](0 until 10000) //https://issues.scala-lang.org/browse/SI-6948
        .toArray
      for (times <- 0 until 10000 / MiniBatchSize) yield {
        val randomIndexArray =
          randomIndex.slice(times * MiniBatchSize, (times + 1) * MiniBatchSize)
        trainData(randomIndexArray)
      }
    }).flatten

  val (lossSeq, trainAccuracySeq, testAccuracySeq) = resultTuple.unzip3

  val plot = Seq(
    Scatter(resultTuple.indices, lossSeq, name = "loss"),
    Scatter(resultTuple.indices, trainAccuracySeq, name = "trainAccuracy"),
    Scatter(resultTuple.indices, testAccuracySeq, name = "testAccuracy")
  )

  plot.plot(title = "loss,trainAccuracy,testAccuracy by time")
}
