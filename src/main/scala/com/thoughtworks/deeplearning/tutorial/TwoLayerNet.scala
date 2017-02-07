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
  * Created by 张志豪 on 2017/2/6.
  */
object TwoLayerNet extends App {

  implicit val optimizerFactory = new DifferentiableINDArray.OptimizerFactory {
    override def ndArrayOptimizer(
        weight: DifferentiableINDArray.Layers.Weight): L2Regularization = {
      new DifferentiableINDArray.Optimizers.L2Regularization {
        override protected def l2Regularization = 0.03

        var learningRate = 0.001

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
  val CLASSES: Int = 10

  //加载测试数据，我们读取100条作为测试数据
  val testNDArray =
    ReadCIFAR10ToNDArray.readFromResource(
      "/cifar-10-batches-bin/test_batch.bin",
      100)

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

  val test_data = testNDArray.head

  val test_expect_result = testNDArray.tail.head

  val test_p = makeVectorized(test_expect_result)

  def fullyConnectedThenRelu(inputSize: Int, outputSize: Int)(
      implicit row: From[INDArray]##T): To[INDArray]##T = {
    val w =
      (Nd4j.randn(inputSize, outputSize) / math.sqrt(outputSize / 2.0)).toWeight * 0.1
    val b = Nd4j.zeros(outputSize).toWeight
    max((row dot w) + b, 0.0)
  }

//  def sigmoid(implicit input: From[INDArray]##T): To[INDArray]##T = {
//    1.0 / (exp(-input) + 1.0)
//  }

  def softmax(implicit scores: From[INDArray]##T): To[INDArray]##T = {
    val expScores = exp(scores)
    expScores / expScores.sum(1)
  }

  def fullyConnectedThenSoftmax(inputSize: Int, outputSize: Int)(
      implicit row: From[INDArray]##T): To[INDArray]##T = {
    val w =
      (Nd4j.randn(inputSize, outputSize) / math.sqrt(outputSize)).toWeight //* 0.1
    val b = Nd4j.zeros(outputSize).toWeight
    //sigmoid.compose((row dot w) + b)
    softmax.compose((row dot w) + b)
  }

  def hiddenLayer(implicit input: From[INDArray]##T): To[INDArray]##T = {
    val layer0 = fullyConnectedThenRelu(3072, 500).compose(input)
//    layer0.withOutputDataHook { data =>
//      println(data)
//    }
    //val layer1 = fullyConnectedThenRelu(3072, 3072).compose(layer0)
    //val layer2 = fullyConnectedThenRelu(3072, 3072).compose(layer1)
    fullyConnectedThenSoftmax(500, 10).compose(layer0)
  }

  val predictor = hiddenLayer

//  implicit def optimizer: Optimizer = new LearningRate {
//    def currentLearningRate() = 0.0001
//  }

  def crossEntropy(
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
    crossEntropy.compose(score :: label :: hnilLayer)
  }

  val trainer = network

  for (_ <- 0 until 2000) {
    val trainNDArray = ReadCIFAR10ToNDArray.getSGDTrainNDArray(256)
    val loss = network.train(
      trainNDArray.head :: makeVectorized(trainNDArray.tail.head) :: HNil)
    println(s"loss : $loss")
  }

  val result = predictor.predict(test_data)
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
