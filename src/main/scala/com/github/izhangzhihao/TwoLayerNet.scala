package com.github.izhangzhihao

import com.thoughtworks.deeplearning.math._
import com.thoughtworks.deeplearning.differentiable.Any._
import com.thoughtworks.deeplearning.differentiable.INDArray.{Optimizer => INDArrayOptimizer, Weight => INDArrayWeight}
import INDArrayOptimizer.{L2Regularization, LearningRate}
import com.github.izhangzhihao.GettingStarted.polyLoss
import com.thoughtworks.deeplearning.Lift
import com.thoughtworks.deeplearning.Tape
import com.thoughtworks.deeplearning.differentiable.INDArray.implicits._
import com.thoughtworks.each.Monadic._
import com.thoughtworks.raii.asynchronous
import com.thoughtworks.raii.asynchronous.Do
import com.thoughtworks.raii.ownership.Borrowing
import org.nd4j.linalg.api.ndarray.INDArray
import com.thoughtworks.deeplearning.differentiable.Double.implicits._
import com.thoughtworks.deeplearning.{Tape, differentiable}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._

import scala.concurrent.ExecutionContext.Implicits.global
import scalaz.concurrent.Task
import scalaz.{-\/, \/, \/-}
import scalaz.std.vector._
import scalaz.std.list._
import scalaz.std.map._
import scalaz.std.iterable._
import plotly.Scatter
import plotly.Plotly._
import shapeless._

import scala.util.Random

/**
  * @author 张志豪 (izhangzhihao) &lt;izhangzhihao@hotmail.com&gt;
  */
object TwoLayerNet extends App {
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

  implicit val optimizerFactory =
    new differentiable.INDArray.OptimizerFactory {
      override def indarrayOptimizer(
          weight: INDArrayWeight): INDArrayOptimizer = {
        new LearningRate with L2Regularization {

          var learningRate = 0.01

          override protected def currentLearningRate(): Double = {
            learningRate *= 0.9995
            learningRate
          }

          override protected def l2Regularization: Double = 0.03
        }
      }
    }

  def fullyConnectedThenRelu(inputSize: Int,
                             outputSize: Int,
                             input: differentiable.INDArray) = {
    //TODO
    val weight =
      (Nd4j.randn(inputSize, outputSize) / math.sqrt(outputSize / 2.0)).toWeight * 0.1
    val b = Nd4j.zeros(outputSize).toWeight
    max(dot(input, weight) + b, 0.0)
  }

  def softmax(scores: differentiable.INDArray): differentiable.INDArray = {
    val expScores = exp(scores)
    expScores / sum(expScores, 1)
  }

  def fullyConnectedThenSoftmax(
      inputSize: Int,
      outputSize: Int,
      input: differentiable.INDArray): differentiable.INDArray = {
    //TODO
    val weight =
      (Nd4j.randn(inputSize, outputSize) / math.sqrt(outputSize)).toWeight
    val b = Nd4j.zeros(outputSize).toWeight
    softmax(dot(input, weight) + b)
  }

  val NumberOfPixels: Int = 3072

  def myNeuralNetwork(
      input: differentiable.INDArray): differentiable.INDArray = {

    val layer0 = fullyConnectedThenRelu(NumberOfPixels, 500, input)

    fullyConnectedThenSoftmax(500, 10, layer0)
  }

  def crossEntropy(score: differentiable.INDArray,
                   label: differentiable.INDArray): differentiable.Double = {
    -mean(
      label * log(score * 0.9 + 0.1) + (1.0 - label) * log(1.0 - score * 0.9))
  }

  def lossFunction(input: INDArray,
                   expectOutput: INDArray)(implicit liftINDArray: Lift.Aux[INDArray, INDArray, INDArray]): differentiable.Double = {

    val score
      : differentiable.INDArray = myNeuralNetwork(1.0 * input) // TODO
    crossEntropy(score, 1.0 * expectOutput) //TODO
  }

  @monadic[Task]
  val trainTask: Task[Unit] = {
    val random = new Random

    val MiniBatchSize = 256

    val lossSeq =
      (
        for (_ <- (0 to 50).toVector) yield {
          val randomIndex = random
            .shuffle[Int, IndexedSeq](0 until 10000) //https://issues.scala-lang.org/browse/SI-6948
            .toArray
          for (times <- (0 until 10000 / MiniBatchSize).toVector) yield {
            val randomIndexArray =
              randomIndex.slice(times * MiniBatchSize,
                                (times + 1) * MiniBatchSize)
            val trainNDArray :: expectLabel :: shapeless.HNil =
              ReadCIFAR10ToNDArray.getSGDTrainNDArray(randomIndexArray)
            val input =
              trainNDArray.reshape(MiniBatchSize, 3072)

            val expectLabelVectorized =
              Utils.makeVectorized(expectLabel, NumberOfClasses)
            val loss = train(lossFunction(input, expectLabelVectorized)).each
            println(loss)
            loss
          }
        }
      ).flatten

    polyLoss(lossSeq)
  }

  val predictResult = throwableMonadic[Task] {
    trainTask.each

    predict(myNeuralNetwork(1.0 * testData)).each //TODO
  }

  predictResult.unsafePerformSyncAttempt match {
    case -\/(e) => throw e
    case \/-(result) =>
      println(Utils.getAccuracy(result, testExpectResult))
  }

}
