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
  Layer
}
import com.thoughtworks.deeplearning.Layer.Batch
import com.thoughtworks.deeplearning.Lift.Layers.Identity
import com.thoughtworks.deeplearning.Lift._
import com.thoughtworks.deeplearning.Poly.MathFunctions._
import com.thoughtworks.deeplearning.Poly.MathMethods./
import com.thoughtworks.deeplearning.Poly.MathOps
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.cpu.nativecpu.NDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.{INDArrayIndex, NDArrayIndex}
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4s.Implicits._
import shapeless._

/**
  * Created by 张志豪 on 2017/1/24.
  * 这段代码是在SoftmaxLinearClassifier基础上修改而来，仅有部分debug的代码做了修改
  */
object Debug extends App {
  //CIFAR10中的图片共有10个分类(airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck)
  val NumberOfClasses: Int = 10

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

  private val train_data = trainNDArray.head
  private val test_data = testNDArray.head

  val train_expect_result = trainNDArray.tail.head
  val test_expect_result = testNDArray.tail.head

  val p = Utils.makeVectorized(train_expect_result, NumberOfClasses)
  val test_p = Utils.makeVectorized(test_expect_result, NumberOfClasses)

  def softmax(implicit scores: From[INDArray]##T): To[INDArray]##T = {
    val expScores: To[INDArray]##T = exp(scores)
      .withOutputDataHook { data =>
        println(data)
      }
    expScores / expScores.sum(1)
  }

  def createMyNeuralNetwork(
      implicit input: From[INDArray]##T): To[INDArray]##T = {
    val initialValueOfWeight = Nd4j.randn(3072, NumberOfClasses) * 0.001
    val weight: To[INDArray]##T = initialValueOfWeight.toWeight
    val result: To[INDArray]##T = input dot weight
    softmax.compose(result) //对结果调用softmax方法，压缩结果值在0到1之间方便处理
  }

  val myNeuralNetwork: FromTo[INDArray, INDArray]##T = createMyNeuralNetwork

  implicit def optimizer: Optimizer = new LearningRate {
    def currentLearningRate() = 0.00001
  }

  def lossFunction(
      implicit pair: From[INDArray :: INDArray :: HNil]##T): To[Double]##T = {
    val input = pair.head
    val expectedOutput = pair.tail.head
    val probabilities = myNeuralNetwork.compose(input)

    -(expectedOutput * log(probabilities)).sum //此处和准备一节中的交叉熵损失对应
  }

  for (_ <- 0 until 2000) {
    val loss = lossFunction.train(train_data :: p :: HNil)
    println(s"loss : $loss")
  }

  private val result = myNeuralNetwork.predict(test_data)
  println(s"result: $result") //输出判断结果

  val right = Utils.getAccuracy(result, test_expect_result)
  println(s"the result is $right %")

}
