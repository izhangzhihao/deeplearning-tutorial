package com.thoughtworks.deeplearning.tutorial

import com.thoughtworks.deeplearning
import org.nd4j.linalg.api.ndarray.INDArray
import com.thoughtworks.deeplearning.DifferentiableHList._
import com.thoughtworks.deeplearning.DifferentiableDouble._
import com.thoughtworks.deeplearning.DifferentiableINDArray._
import com.thoughtworks.deeplearning.DifferentiableAny._
import com.thoughtworks.deeplearning.DifferentiableINDArray.Optimizers._
import com.thoughtworks.deeplearning.{DifferentiableHList, DifferentiableINDArray, Layer}
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
  * Created by 张志豪 on 2017/1/22.
  */
object MiniBatchGradientDescent extends App {
  //CIFAR10中的图片共有10个分类(airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck)
  val CLASSES: Int = 10

  //加载测试数据，我们读取100条作为测试数据
  val testNDArray =
    ReadCIFAR10ToNDArray.readFromResource("/cifar-10-batches-bin/test_batch.bin", 100)

  /**
    * 处理标签数据：将N行一列的NDArray转换为N行CLASSES列的NDArray，每行对应的正确分类的值为1，其它列的值为0
    *
    * @param ndArray 标签数据
    * @return N行CLASSES列的NDArray
    */
  def makeVectorized(ndArray: INDArray): INDArray = {
    val shape = ndArray.shape()

    val p = Nd4j.zeros(shape(0), CLASSES)
    for (i <- 0 until shape(0)) {
      val double = ndArray.getDouble(i, 0)
      val column = double.toInt
      p.put(i, column, 1)
    }
    p
  }

  private val test_data = testNDArray.head

  val test_expect_result = testNDArray.tail.head

  val test_p = makeVectorized(test_expect_result)

  def softmax(implicit scores: From[INDArray] ## T): To[INDArray] ## T = {
    val expScores = exp(scores)
    expScores / expScores.sum(1)
  }

  def createMyNeuralNetwork(implicit input: From[INDArray] ## T): To[INDArray] ## T = {
    val initialValueOfWeight = Nd4j.randn(3072, CLASSES) * 0.001
    val weight: To[INDArray] ## T = initialValueOfWeight.toWeight
    val result: To[INDArray] ## T = input dot weight
    softmax.compose(result) //对结果调用softmax方法，压缩结果值在0到1之间方便处理
  }

  val myNeuralNetwork: FromTo[INDArray, INDArray] ## T = createMyNeuralNetwork

  implicit def optimizer: Optimizer = new LearningRate {
    def currentLearningRate() = 0.00001
  }

  def lossFunction(implicit pair: From[INDArray :: INDArray :: HNil] ## T): To[Double] ## T = {
    val input = pair.head
    val expectedOutput = pair.tail.head
    val probabilities = myNeuralNetwork.compose(input)

    -(expectedOutput * log(probabilities)).sum //此处和准备一节中的交叉熵损失对应
  }

  for (_ <- 0 until 2000) {
    val trainNDArray = ReadCIFAR10ToNDArray.getSGDTrainNDArray(256)
    val loss = lossFunction.train(trainNDArray.head :: makeVectorized(trainNDArray.tail.head) :: HNil)
    println(s"loss : $loss")
  }

  private val result = myNeuralNetwork.predict(test_data)
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

  private val shape = result.shape()
  for (row <- 0 until shape(0)) {
    val rowItem = result.getRow(row)
    val index = findMaxItemIndex(rowItem)
    if (index == test_expect_result.getDouble(row, 0)) {
      right += 1
    }
  }
  println(s"the result is $right %")
}
