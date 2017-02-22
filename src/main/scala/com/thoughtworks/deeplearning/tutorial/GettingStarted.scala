package com.thoughtworks.deeplearning.tutorial

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
import plotly.Scatter
import shapeless._
import plotly.Plotly._
import plotly._

/**
  * Created by 张志豪 on 1/16/17.
  */
object GettingStarted extends App {

  val myNeuralNetwork: (INDArray <=> INDArray)##T = createMyNeuralNetwork

  val input: INDArray =
    Array(Array(0, 1, 2), Array(3, 6, 9), Array(13, 15, 17)).toNDArray
  val predictionResult: INDArray = myNeuralNetwork.predict(input)

  def createMyNeuralNetwork(
      implicit input: From[INDArray]##T): To[INDArray]##T = {
    val initialValueOfWeight = Nd4j.randn(3, 1)
    val weight: To[INDArray]##T = initialValueOfWeight.toWeight
    input dot weight
  }

  implicit def optimizer: Optimizer = new LearningRate {
    def currentLearningRate() = 0.001
  }

  def lossFunction(
      implicit pair: From[INDArray :: INDArray :: HNil]##T): To[Double]##T = {
    val input = pair.head
    val expectedOutput = pair.tail.head
    abs(myNeuralNetwork.compose(input) - expectedOutput).sum
  }

  val expectedOutput: INDArray = Array(Array(1), Array(3), Array(2)).toNDArray

  val lossSeq = for (_ <- 0 until 30) yield {
    val loss = lossFunction.train(input :: expectedOutput :: HNil)
    println(loss)
    loss
  }

  val plot = Seq(
    Scatter(lossSeq.indices, lossSeq)
  )

  plot.plot(
    title = "loss on time"
  )

  // The loss should be close to zero
  println(s"loss: ${lossFunction.predict(input :: expectedOutput :: HNil)}")

  // The prediction result should be close to expectedOutput
  println(s"result: ${myNeuralNetwork.predict(input)}")

}
