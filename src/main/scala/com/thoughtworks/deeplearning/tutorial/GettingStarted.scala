package com.thoughtworks.deeplearning.tutorial

import com.thoughtworks.deeplearning.math._
import com.thoughtworks.deeplearning.differentiable.Any._
import com.thoughtworks.deeplearning.differentiable.INDArray.Optimizer
import com.thoughtworks.deeplearning.differentiable.INDArray.Optimizer.LearningRate
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
//import scalaz.std.option._
import scalaz.{-\/, \/, \/-}
//import scalaz.syntax.all._
//import scalaz.std.iterable._
import scalaz.std.vector._

import plotly.Scatter
import plotly.Plotly._

object GettingStarted extends App {

  implicit def optimizer: Optimizer = new LearningRate {
    def currentLearningRate() = 0.001
  }

  val weight = (Nd4j.randn(3, 1) / scala.math.sqrt(3.0)).toWeight

  def myNeuralNetwork(input: INDArray): differentiable.INDArray = {
    dot(input, weight)
  }

  def lossFunction(input: INDArray,
                   expectOutput: INDArray): differentiable.Double = {
    sumT(abs(myNeuralNetwork(input) - expectOutput))
  }

  def trainMyNetwork(input: INDArray, expectedOutput: INDArray): Task[Double] = {
    train(lossFunction(input, expectedOutput))
  }

  val input: INDArray =
    Array(Array(0, 1, 2), Array(3, 6, 9), Array(13, 15, 17)).toNDArray

  val expectedOutput: INDArray = Array(Array(1), Array(3), Array(2)).toNDArray

  @monadic[Task]
  val trainTask: Task[Unit] = {

    val lossSeq = for (_ <- (1 to 400).toVector) yield {
      trainMyNetwork(input, expectedOutput).each
    }

    polyLoss(lossSeq)

  }

  def polyLoss(lossSeq: IndexedSeq[Double]) = {
    val plot = Seq(
      Scatter(lossSeq.indices, lossSeq)
    )

    plot.plot(
      title = "loss by time"
    )
  }

  val predictResult = throwableMonadic[Task] {
    trainTask.each
    predict(myNeuralNetwork(input)).each
  }

  predictResult.unsafePerformSyncAttempt match {
    case -\/(e) => throw e
    case \/-(result) =>
      println(result)
  }

}
