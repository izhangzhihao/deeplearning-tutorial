package com.thoughtworks.deeplearning.tutorial

import com.thoughtworks.deeplearning
import org.nd4j.linalg.api.ndarray.INDArray
import com.thoughtworks.deeplearning.DifferentiableHList._
import com.thoughtworks.deeplearning.DifferentiableDouble._
import com.thoughtworks.deeplearning.DifferentiableINDArray._
import com.thoughtworks.deeplearning.DifferentiableAny._
import com.thoughtworks.deeplearning.DifferentiableINDArray.Layers.Weight
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
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.{INDArrayIndex, NDArrayIndex}
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4s.Implicits._
import plotly.Scatter
import shapeless._
import org.nd4s.Implicits._
import plotly.Scatter
import shapeless._
import plotly.Plotly._
import plotly._

import scala.util.Random

/**
  * Created by 张志豪 on 2017/2/6.
  */
object TwoLayerNet extends App {

  implicit val optimizerFactory = new DifferentiableINDArray.OptimizerFactory {
    override def ndArrayOptimizer(weight: Weight): Optimizer = {
      new LearningRate with L2Regularization {

        var learningRate = 0.001

        override protected def currentLearningRate(): Double = {
          learningRate *= 0.9995
          learningRate
        }

        override protected def l2Regularization: Double = 0.03
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

  val testData = testNDArray.head

  val testExpectResult = testNDArray.tail.head

  val vectorizedTestExpectResult =
    Utils.makeVectorized(testExpectResult, NumberOfClasses)

  def fullyConnectedThenRelu(inputSize: Int, outputSize: Int)(
      implicit row: INDArray @Symbolic): INDArray @Symbolic = {
    val w =
      (Nd4j.randn(inputSize, outputSize) / math.sqrt(outputSize / 2.0)).toWeight * 0.1
    val b = Nd4j.zeros(outputSize).toWeight
    max((row dot w) + b, 0.0)
  }

  def softmax(implicit scores: INDArray @Symbolic): INDArray @Symbolic = {
    val expScores = exp(scores)
    expScores / expScores.sum(1)
  }

  def fullyConnectedThenSoftmax(inputSize: Int, outputSize: Int)(
      implicit row: INDArray @Symbolic): INDArray @Symbolic = {
    val w =
      (Nd4j.randn(inputSize, outputSize) / math.sqrt(outputSize)).toWeight //* 0.1
    val b = Nd4j.zeros(outputSize).toWeight
    softmax.compose((row dot w) + b)
  }

  def hiddenLayer(implicit input: INDArray @Symbolic): INDArray @Symbolic = {
    val layer0 = fullyConnectedThenRelu(3072, 500).compose(input)
    fullyConnectedThenSoftmax(500, 10).compose(layer0)
  }

  val predictor = hiddenLayer

  def crossEntropy(implicit pair: (INDArray :: INDArray :: HNil) @Symbolic)
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
    val hnilLayer: HNil @Symbolic = HNil
    crossEntropy.compose(score :: label :: hnilLayer)
  }

  val trainer = network

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
          val loss = trainer.train(input :: expectLabelVectorized :: HNil)
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

  val result = predictor.predict(testData)
  println(s"result: $result") //输出判断结果

  val right = Utils.getAccuracy(result, testExpectResult)
  println(s"the result is $right %")
}
