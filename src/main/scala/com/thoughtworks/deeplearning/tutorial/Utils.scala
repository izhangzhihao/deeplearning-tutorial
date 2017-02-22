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
    (0 until col).foreach { index =>
      {
        val itemValue = iNDArray.getDouble(0, index)
        if (itemValue > maxValue) {
          maxValue = itemValue
          maxIndex = index
        }
      }
    }
    maxIndex
  }

  /**
    * 计算准确率
    * @param result 预测结果
    * @param test_expect_result 期望结果
    * @return 准确率
    */
  def getAccuracy(result: INDArray, test_expect_result: INDArray): Double = {
    var right = 0.0
    val shape = result.shape()
    for (row <- 0 until shape(0)) {
      val rowItem = result.getRow(row)
      val index = findMaxItemIndex(rowItem)
      if (index == test_expect_result.getDouble(row, 0)) {
        right += 1.0
      }
    }
    right
  }
}
