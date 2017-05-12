package com.github.izhangzhihao

import com.thoughtworks.deeplearning.math._
import com.thoughtworks.deeplearning.differentiable.Any._
import com.thoughtworks.deeplearning.differentiable.INDArray.{
  Optimizer => INDArrayOptimizer
}
import INDArrayOptimizer.LearningRate
import com.github.izhangzhihao.GettingStarted.{
  expectedOutput,
  input,
  lossFunction,
  myNeuralNetwork,
  polyLoss,
  trainTask
}
import com.thoughtworks.deeplearning.differentiable.INDArray.implicits._
import com.thoughtworks.each.Monadic._
import com.thoughtworks.raii.asynchronous.Do
import com.thoughtworks.deeplearning.differentiable.Double._
import com.thoughtworks.deeplearning.differentiable.Double.implicits._
import com.thoughtworks.deeplearning.{Tape, differentiable}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._

import scala.concurrent.ExecutionContext.Implicits.global
import scalaz.concurrent.Task
import scalaz.{-\/, \/, \/-}
import scalaz.std.vector._
import plotly.Scatter
import plotly.Plotly._

/**
  * @author 张志豪 (izhangzhihao) &lt;izhangzhihao@hotmail.com&gt;
  */
object SoftmaxLinearClassifier extends App {
  //CIFAR10中的图片共有10个分类(airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck)
  val NumberOfClasses: Int = 10
  val NumberOfPixels: Int = 3072

  //加载train数据,我们读取1000条数据作为训练数据
  val trainNDArray =
    ReadCIFAR10ToNDArray.readFromResource(
      "/cifar-10-batches-bin/data_batch_1.bin",
      1000)

  //加载测试数据，我们读取100条作为测试数据
  val testNDArray =
    ReadCIFAR10ToNDArray.readFromResource(
      "/cifar-10-batches-bin/test_batch.bin",
      100)

  val trainData = trainNDArray.head
  val testData = testNDArray.head

  val trainExpectResult = trainNDArray.tail.head
  val testExpectResult = testNDArray.tail.head

  val vectorizedTrainExpectResult =
    Utils.makeVectorized(trainExpectResult, NumberOfClasses)
  val vectorizedTestExpectResult =
    Utils.makeVectorized(testExpectResult, NumberOfClasses)

  def softmax(scores: differentiable.INDArray): differentiable.INDArray = {
    val expScores = exp(scores)
    expScores / sum(expScores, 1)
  }

  implicit def optimizer: INDArrayOptimizer = new LearningRate {
    def currentLearningRate() = 0.00001
  }

  val weight: differentiable.INDArray =
    (Nd4j.randn(NumberOfPixels, NumberOfClasses) * 0.001).toWeight

  def myNeuralNetwork(input: INDArray): differentiable.INDArray = {
    softmax(dot(input, weight))
  }

  def lossFunction(input: INDArray,
                   expectOutput: INDArray): differentiable.Double = {
    val probabilities = myNeuralNetwork(input)
    -mean(log(probabilities) * expectOutput)
  }

  @monadic[Task]
  val trainTask: Task[Unit] = {

    val lossSeq = for (_ <- (1 to 2000).toVector) yield {
      val loss =
        train(lossFunction(trainData, vectorizedTrainExpectResult)).each
      println(loss)
      loss
    }

    polyLoss(lossSeq)
  }

  val predictResult = throwableMonadic[Task] {
    trainTask.each
    predict(myNeuralNetwork(testData)).each
  }

  predictResult.unsafePerformSyncAttempt match {
    case -\/(e) => throw e
    case \/-(result) =>
      println(Utils.getAccuracy(result, testExpectResult))
  }

}
