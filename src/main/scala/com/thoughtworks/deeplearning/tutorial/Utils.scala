package com.thoughtworks.deeplearning.tutorial

import com.thoughtworks.deeplearning.DifferentiableINDArray.{
  INDArrayPlaceholder,
  INDArraySemigroupBatch
}
import com.thoughtworks.deeplearning.Layer.Batch
import com.thoughtworks.deeplearning.{BufferedLayer, Layer}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import scala.collection.GenTraversable
import scala.collection.generic.GenericTraversableTemplate
import scala.reflect.ClassTag

/**
  * Created by 张志豪 on 2017/2/22.
  */
object Utils {

  /**
    * 处理标签数据：将N行一列的NDArray转换为N行NumberOfClasses列的NDArray，每行对应的正确分类的值为1，其它列的值为0
    *
    * @param ndArray 标签数据
    * @return N行NumberOfClasses列的NDArray
    */
  def makeVectorized(ndArray: INDArray, numberOfClasses: Int): INDArray = {
    val shape = ndArray.shape()

    val p = Nd4j.zeros(shape(0), numberOfClasses)
    for (i <- 0 until shape(0)) {
      val double = ndArray.getDouble(i, 0)
      val column = double.toInt
      p.put(i, column, 1)
    }
    p
  }

  /**
    * 二维INDArray中获得值最大的元素所在的列组成的INDArray
    *
    * @param iNDArray
    * @return
    */
  def findMaxItemIndex(iNDArray: INDArray): INDArray = {
    Nd4j.argMax(iNDArray, 1)
  }

  /**
    * 计算准确率
    * @param result 预测结果
    * @param test_expect_result 期望结果
    * @return 准确率
    */
  def getAccuracy(result: INDArray, test_expect_result: INDArray): Double = {

    assert(test_expect_result.shape().toSeq.last == 1)

    val iNDArrayIndex = findMaxItemIndex(result)
    val acc = for (row <- 0 until iNDArrayIndex.shape()(0)) yield {
      if (iNDArrayIndex.getDouble(row, 0) ==
            test_expect_result.getDouble(row, 0)) {
        1.0
      } else 0.0
    }
    (acc.sum / result.shape()(0)) * 100
  }

  final case class GetAccuracy[Input0 <: Batch](
      operand1: Layer.Aux[Input0, INDArrayPlaceholder.Batch],
      operand2: Layer.Aux[Input0, INDArrayPlaceholder.Batch]
  ) extends BufferedLayer.Binary {

    type BufferedBatch =
      INDArraySemigroupBatch with SemigroupBatch with BinaryBatch

    type Input = Input0

    override protected def rawForward(input0: Input0): BufferedBatch = {
      new {
        override final val input = input0
      } with INDArraySemigroupBatch with SemigroupBatch with BinaryBatch {

        val value = ???

        override protected def rawBackward(outputDelta: INDArray): Unit = {
          ???
        }
      }
    }
  }

//  final class INDArrayLayerOps[Input <: Batch](
//      operand: Layer.Aux[Input, INDArrayPlaceholder.Batch]) {
//    def getAccuracy(right: Layer.Aux[Input, INDArrayPlaceholder.Batch])
//      : Layer.Aux[Input, DoublePlaceholder.Batch] = {
//      GetAccuracy(operand, right)
//    }
//  }

  class Unzipper[A, CC[X] <: GenTraversable[X]](
      s: GenericTraversableTemplate[A, CC]) {
    def unzip4[A1, A2, A3, A4](implicit asQuad: A => (A1, A2, A3, A4))
      : (CC[A1], CC[A2], CC[A3], CC[A4]) = {
      val b1 = s.genericBuilder[A1]
      val b2 = s.genericBuilder[A2]
      val b3 = s.genericBuilder[A3]
      val b4 = s.genericBuilder[A4]
      for (e <- s) {
        val (a, b, c, d) = asQuad(e)
        b1 += a
        b2 += b
        b3 += c
        b4 += d
      }
      (b1.result, b2.result, b3.result, b4.result)
    }
  }

  implicit def toUnzipper[A, CC[X] <: GenTraversable[X]](
      s: GenericTraversableTemplate[A, CC]): Unzipper[A, CC] = new Unzipper(s)

  class Tuple2ToArray[A: ClassTag](tuple: (A, A)) {
    def toArray: Array[A] = {
      val (one: A, two: A) = tuple
      Array(one, two)
    }
  }

  implicit def tupleTwoToArray[A: ClassTag](tuple: (A, A)): Tuple2ToArray[A] =
    new Tuple2ToArray[A](tuple)

}
