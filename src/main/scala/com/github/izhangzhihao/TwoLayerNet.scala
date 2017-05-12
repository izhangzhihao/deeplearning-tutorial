package com.github.izhangzhihao

import com.thoughtworks.deeplearning.math._
import com.thoughtworks.deeplearning.differentiable.Any._
import com.thoughtworks.deeplearning.differentiable.INDArray.{
  Optimizer => INDArrayOptimizer,
  Weight => INDArrayWeight
}
import INDArrayOptimizer.{L2Regularization, LearningRate}
import com.github.izhangzhihao.GettingStarted.polyLoss
import com.thoughtworks.deeplearning.differentiable.INDArray.implicits._
import com.thoughtworks.each.Monadic._
import com.thoughtworks.raii.asynchronous.Do
//import com.thoughtworks.deeplearning.differentiable.Double._
import com.thoughtworks.deeplearning.differentiable.Double.implicits._
import com.thoughtworks.deeplearning.differentiable.INDArray
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
  val NumberOfPixels: Int = 3072

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
    val weight =
      (Nd4j.randn(inputSize, outputSize) / math.sqrt(outputSize / 2.0)).toWeight * 0.1
    val b = Nd4j.zeros(outputSize).toWeight
    max(dot(input, weight) + b, 0.0)
  }
}
