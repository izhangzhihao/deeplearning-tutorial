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
import com.thoughtworks.deeplearning._
import com.thoughtworks.deeplearning.Layer.Batch
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

        var learningRate = 0.000001

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

  /**
    * 处理标签数据：将N行一列的NDArray转换为N行NumberOfClasses列的NDArray，每行对应的正确分类的值为1，其它列的值为0
    *
    * @param ndArray 标签数据
    * @return N行NumberOfClasses列的NDArray
    */
  def makeVectorized(ndArray: INDArray): INDArray = {
    val shape = ndArray.shape()

    val p = Nd4j.zeros(shape(0), NumberOfClasses)
    for (i <- 0 until shape(0)) {
      val double = ndArray.getDouble(i, 0)
      val column = double.toInt
      p.put(i, column, 1)
    }
    p
  }

  val test_data = testNDArray.head

  val test_expect_result = testNDArray.tail.head

  val test_p = makeVectorized(test_expect_result)

  val MiniBatchSize = 4

  val Depth = 3

  val InputSize = 32 // W 输入数据尺寸

  val KernelNumber = 3 //卷积核的数量

  def convolutionThenRelu(implicit input: From[INDArray]##T): To[INDArray]##T = {

    val imageCount = input.shape(0)
    val inputSize = input.shape(2)

    val Stride = 1 // 步长

    val Padding = 1 //零填充数量

    val KernelSize = 3 //F 卷积核的空间尺寸

    val outputSize = inputSize

    val weight =
      (Nd4j.randn(Array(KernelNumber, Depth, KernelSize, KernelSize)) /
        math.sqrt(KernelSize / 2.0)).toWeight * 0.1
    val bias = Nd4j.zeros(KernelNumber).toWeight

    val colRow = input.im2col(Array(KernelSize, KernelSize),
                              Array(Stride, Stride),
                              Array(Padding, Padding))

    val permuteCol = colRow.permute(0, 2, 3, 1, 4, 5)

    val col2dRow = permuteCol.reshape(
      imageCount * outputSize * outputSize,
      (Depth * KernelSize * KernelSize).toLayer)

    val reshapedW =
      weight.reshape(KernelSize * KernelSize * Depth, KernelNumber)

    val res = max((col2dRow dot reshapedW) + bias, 0.0)

    val result =
      res.reshape(imageCount, outputSize, outputSize, KernelNumber.toLayer)

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

    val w =
      (Nd4j.randn(inputSize, outputSize) / math.sqrt(outputSize)).toWeight
    val b = Nd4j.zeros(outputSize).toWeight
    softmax.compose((input.reshape(imageCount, inputSize.toLayer) dot w) + b)
  }

  def hiddenLayer(implicit input: From[INDArray]##T): To[INDArray]##T = {
    val layer0 = convolutionThenRelu.compose(input)
    val layer1 = convolutionThenRelu.compose(layer0)
    val layer2 = maxPool.compose(layer1)

//    val layer3 = convolutionThenRelu.compose(layer2)
//    val layer4 = convolutionThenRelu.compose(layer3)
//    val layer5 = maxPool.compose(layer4)

    fullyConnectedThenSoftmax(3 * 16 * 16, 10).compose(layer2)
//    fullyConnectedThenSoftmax(3 * 32 * 32, 10).compose(layer0)

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

//
//  val random = new util.Random
//
//  for (epic <- 0 until 5) {
//    val randomIndex = random
//      .shuffle[Int, IndexedSeq](0 until 10000) //https://issues.scala-lang.org/browse/SI-6948
//      .toArray
//    for (times <- 0 until 10000 / MiniBatchSize) {
//      val thisindex =
//        randomIndex.slice(times * MiniBatchSize, (times + 1) * MiniBatchSize)
//      //train>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//    }
//  }

  def assertClear(layer: Any): Unit = {
    layer match {
      case cached: BufferedLayer =>
        assert(cached.cache.isEmpty)
      case _ =>
    }
    layer match {
      case parent: Product =>
        for (upstreamLayer <- parent.productIterator) {
          assertClear(upstreamLayer)
        }
      case _ =>
    }
  }

  var lossSeq =
    for (_ <- 0 until 1000) yield {
      val trainNDArray :: label :: HNil =
        ReadCIFAR10ToNDArray.getSGDTrainNDArray(MiniBatchSize)
      val input =
        trainNDArray.reshape(MiniBatchSize, Depth, InputSize, InputSize)
      val expectResult = makeVectorized(label)

      assertClear(predictor)
      val loss = trainNetwork.train(input :: expectResult :: HNil)
      assertClear(predictor)

      println(s"loss : $loss")
      loss
    }
  val plot = Seq(
    Scatter(
      0 until 1000 by 1,
      lossSeq
    )
  )

  plot.plot(
    title = "loss on time"
  )

  val result =
    predictor.predict(test_data.reshape(100, Depth, InputSize, InputSize))
  println(s"result: $result") //输出判断结果

  /**
    * 从一行INDArray中获得值最大的元素所在的列
    *
    * @param iNDArray
    * @return
    */
  def findMaxItemIndex(iNDArray: INDArray): Int = {
    val shape = iNDArray.shape()
    val col = shape(1)
    var maxValue = 0.0
    var maxIndex = 0
    for (index <- 0 until col) {
      val itemValue = iNDArray.getDouble(0, index)
      if (itemValue > maxValue) {
        maxValue = itemValue
        maxIndex = index
      }
    }
    maxIndex
  }

  var right = 0

  val shape = result.shape()
  for (row <- 0 until shape(0)) {
    val rowItem = result.getRow(row)
    val index = findMaxItemIndex(rowItem)
    if (index == test_expect_result.getDouble(row, 0)) {
      right += 1
    }
  }
  println(s"the result is $right %")
}
