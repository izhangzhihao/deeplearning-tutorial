package com.thoughtworks.deeplearning.tutorial

import com.thoughtworks.deeplearning
import org.nd4j.linalg.api.ndarray.INDArray
import com.thoughtworks.deeplearning.DifferentiableHList._
import com.thoughtworks.deeplearning.DifferentiableDouble._
import com.thoughtworks.deeplearning.DifferentiableINDArray._
import com.thoughtworks.deeplearning.DifferentiableAny._
import com.thoughtworks.deeplearning.DifferentiableINDArray.Optimizers._
import com.thoughtworks.deeplearning.{
  DifferentiableHList,
  DifferentiableINDArray,
  Layer,
  Symbolic
}
import com.thoughtworks.deeplearning.Layer.Tape
import com.thoughtworks.deeplearning.Symbolic.Layers.Identity
import com.thoughtworks.deeplearning.Symbolic._
import com.thoughtworks.deeplearning.Poly.MathFunctions._
import com.thoughtworks.deeplearning.Poly.MathMethods./
import com.thoughtworks.deeplearning.Poly.MathOps
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.cpu.nativecpu.NDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.{INDArrayIndex, NDArrayIndex}
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4s.Implicits._
import plotly.Scatter
import shapeless._
import plotly.Plotly._
import plotly._

import scala.util.Random

/**
  * Created by 张志豪 on 2017/1/22.
  */
object MiniBatchGradientDescent extends App {
  //CIFAR10中的图片共有10个分类(airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck)
  val NumberOfClasses: Int = 10

  //加载测试数据，我们读取100条作为测试数据
  val testNDArray =
    ReadCIFAR10ToNDArray.readFromResource(
      "/cifar-10-batches-bin/test_batch.bin",
      100)

  val testData = testNDArray.head

  val testExpectResult = testNDArray.tail.head

  val vectorizedTestExpectResult =
    Utils.makeVectorized(testExpectResult, NumberOfClasses)

  def softmax(implicit scores: INDArray @Symbolic): INDArray @Symbolic = {
    val expScores = exp(scores)
    expScores / expScores.sum(1)
  }

  def createMyNeuralNetwork(
      implicit input: INDArray @Symbolic): INDArray @Symbolic = {
    val initialValueOfWeight = Nd4j.randn(3072, NumberOfClasses) * 0.001
    val weight: INDArray @Symbolic = initialValueOfWeight.toWeight
    val result: INDArray @Symbolic = input dot weight
    softmax.compose(result) //对结果调用softmax方法，压缩结果值在0到1之间方便处理
  }

  val myNeuralNetwork = createMyNeuralNetwork

  implicit def optimizer: Optimizer = new LearningRate {
    def currentLearningRate() = 0.00001
  }

  def lossFunction(implicit pair: (INDArray :: INDArray :: HNil) @Symbolic)
    : Double @Symbolic = {
    val input = pair.head
    val expectedOutput = pair.tail.head
    val probabilities = myNeuralNetwork.compose(input)

    -(expectedOutput * log(probabilities)).mean //此处和准备一节中的交叉熵损失对应
  }

  val random = new Random

  val MiniBatchSize = 256

  val lossSeq =
    (
      for (_ <- 0 to 50) yield {
        val randomIndex = random
          .shuffle[Int, IndexedSeq](0 until 10000) //https://issues.scala-lang.org/browse/SI-6948
          .toArray
        for (times <- 0 until 10000 / MiniBatchSize) yield {
          val randomIndexArray =
            randomIndex.slice(times * MiniBatchSize,
                              (times + 1) * MiniBatchSize)
          val trainNDArray :: expectLabel :: shapeless.HNil =
            ReadCIFAR10ToNDArray.getSGDTrainNDArray(randomIndexArray)
          val input =
            trainNDArray.reshape(MiniBatchSize, 3072)

          val expectLabelVectorized =
            Utils.makeVectorized(expectLabel, NumberOfClasses)
          val loss = lossFunction.train(input :: expectLabelVectorized :: HNil)
          println(loss)
          loss
        }
      }
    ).flatten

  val plot = Seq(
    Scatter(lossSeq.indices, lossSeq)
  )

  plot.plot(
    title = "loss by time"
  )

  val result = myNeuralNetwork.predict(testData)
  println(s"result: $result") //输出判断结果

  val right = Utils.getAccuracy(result, testExpectResult)
  println(s"the result is $right %")
}
