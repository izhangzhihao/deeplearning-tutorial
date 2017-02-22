package com.thoughtworks.deeplearning.tutorial

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

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

    val iNDArrayIndex = findMaxItemIndex(result)
    val shape = iNDArrayIndex.shape
    val acc = for (row <- 0 until shape(0)) yield {
      if (iNDArrayIndex.getDouble(row, 0) ==
            test_expect_result.getDouble(row, 0)) {
        1.0
      } else 0.0
    }
    acc.sum
  }
}
