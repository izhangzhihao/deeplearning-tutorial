package com.thoughtworks.deeplearning.tutorial

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.convolution.Convolution
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.factory.Nd4j.PadMode
import org.nd4s.Implicits._
import shapeless.{::, HNil}

/**
  * Created by zhangzhihao on 2017/2/10.
  */
object convolutionTest extends App {
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

  val DEPTH = 3

  val MINI_BATCH_SIZE = 100

  val STRIDE = 1 // 步长

  val PADDING = 1 //零填充数量

  val FILTER_NUMBER = 3 //滤波器的数量

  val FILTER_SIZE = 3 //F 滤波器的空间尺寸

  val INPUT_SIZE = 32 // W 输入数据尺寸

  val POOL_SIZE = 2 //maxPool 2*2

  val W1 =
    (Nd4j.randn(Array(FILTER_NUMBER, DEPTH, FILTER_SIZE, FILTER_SIZE)) / math
      .sqrt(FILTER_SIZE / 2.0)) * 0.1

  val B1 = Nd4j.zeros(FILTER_NUMBER)

  val outputSize = (INPUT_SIZE + 2 * PADDING - FILTER_SIZE) / STRIDE + 1

  val output =
    Nd4j.zeros(MINI_BATCH_SIZE, FILTER_NUMBER, outputSize, outputSize)

  val outputShape = output.shape()

  def conv_forward(
      inputData: INDArray :: INDArray :: HNil): (INDArray, INDArray) = {
    val input = inputData.head
    assert((INPUT_SIZE + 2 * PADDING - FILTER_SIZE) % STRIDE == 0)

    assert(outputSize == INPUT_SIZE)

    val inputShape = input.shape() //imageCount*3072

    val imageCount = inputShape(0) //imageCount

    val col = Nd4j.createUninitialized(Array(imageCount,
                                             DEPTH,
                                             INPUT_SIZE + 2 * PADDING,
                                             INPUT_SIZE + 2 * PADDING,
                                             FILTER_SIZE,
                                             FILTER_SIZE),
                                       'c')

    val cols = Convolution.im2col(padImageData,
                                  FILTER_SIZE,
                                  FILTER_SIZE,
                                  STRIDE,
                                  STRIDE,
                                  PADDING,
                                  PADDING,
                                  true,
                                  col) // imageCount*3*34*34*3*3

    val cols2d = cols.reshape(
      'c',
      imageCount * (INPUT_SIZE + 2 * PADDING) * (INPUT_SIZE + 2 * PADDING),
      DEPTH * FILTER_SIZE * FILTER_SIZE) //115600*27

    val reshapeW = W1.reshape(FILTER_SIZE * FILTER_SIZE * DEPTH, -1) //27*3

    //val result = Nd4j.tensorMmul(cols2d, reshapeW, Array(Array(1), Array(1))) //115600*3

    val result = cols2d.dot(reshapeW) //115600*3

    val res = result.addRowVector(B1)

    val out = res.reshape(imageCount,
                          FILTER_NUMBER,
                          INPUT_SIZE + 2 * PADDING,
                          INPUT_SIZE + 2 * PADDING) //imageCount*3*34*34
    (out, cols2d)
  }

  def conv_backword(outputData: INDArray :: INDArray :: HNil,
                    cols2d: INDArray): (INDArray, INDArray, INDArray) = {
    val db = W1.sum(0, 2, 3)
    val dout = W1.permute(1, 2, 3, 0).reshape(FILTER_NUMBER, -1) //3*27

    val dw = dout
      .dot(cols2d.T)
      .reshape(FILTER_NUMBER, DEPTH, FILTER_SIZE, FILTER_SIZE)

    val dx_cols = W1.reshape(FILTER_NUMBER, -1).T.dot(dout)

    val dx = Convolution.im2col(dx_cols,
                                FILTER_SIZE,
                                FILTER_SIZE,
                                STRIDE,
                                STRIDE,
                                PADDING,
                                PADDING,
                                true)
    (dx, dw, db)
  }

  def maxpool_forward(inputData: INDArray :: INDArray :: HNil) = {
    ???
  }

  def maxpool_backword(outputData: INDArray :: INDArray :: HNil) = {
    ???
  }

  def network(inputData: INDArray :: INDArray :: HNil): Double = {

    val out_cols = conv_forward(inputData)

    //W1.sumNumber()

    conv_backword(inputData, out_cols._2)
//    maxpool_forward()
//    maxpool_backword()
//    conv_backword()

    //softmax_loss(out...)

    //每个图片
    /*for (index <- 0 until imageCount) {
      //遍历每个核(使用某一个核来卷积处理图像数据)
      for (filterNumber <- 0 until FILTER_NUMBER) {
        //遍历经过某一个核卷积后输出样本矩阵的长度
        for (outputXSize <- 0 until outputSize by STRIDE) {
          //遍历经过某一个核卷积后输出样本矩阵的宽度
          for (outputYSize <- 0 until outputSize by STRIDE) {
            //遍历每个channel(使用某个核来卷积处理每个channel数据)
            for (depth <- 0 until DEPTH) {
              val w = W.get(NDArrayIndex.point(filterNumber),
                            NDArrayIndex.point(depth),
                            NDArrayIndex.all(),
                            NDArrayIndex.all())
              val x = padImageData.get(
                NDArrayIndex.point(index),
                NDArrayIndex.point(depth),
                NDArrayIndex.interval(outputXSize, FILTER_SIZE + outputXSize),
                NDArrayIndex.interval(outputYSize, FILTER_SIZE + outputYSize)
              )

              val oldOutput = output.get(NDArrayIndex.point(index),
                                         NDArrayIndex.point(filterNumber),
                                         NDArrayIndex.point(outputXSize),
                                         NDArrayIndex.point(outputYSize))
              val newOutput =
                oldOutput + w.dot(x).sumNumber() + B.getDouble(filterNumber)
              output.put(
                Array(NDArrayIndex.point(index),
                      NDArrayIndex.point(filterNumber),
                      NDArrayIndex.point(outputXSize),
                      NDArrayIndex.point(outputYSize)),
                newOutput //激活函数处理？
              )
            }
          }
        }
      }
    }*/

    1.1
  }

  def softmax_loss(inputData: INDArray, label: INDArray): (Double, INDArray) =
    ???

  val trainNDArray = ReadCIFAR10ToNDArray.readFromResource(
    "/cifar-10-batches-bin/test_batch.bin",
    100) //ReadCIFAR10ToNDArray.getSGDTrainNDArray(256)

  val inputImageData =
    trainNDArray.head.reshape(MINI_BATCH_SIZE, DEPTH, INPUT_SIZE, INPUT_SIZE) //imageCount*3*32*32

  val padImageData = Nd4j.pad(
    inputImageData,
    Array(Array(0, 0), Array(0, 0), Array(1, 1), Array(1, 1)),
    PadMode.CONSTANT)

  val loss = network(
    padImageData :: makeVectorized(trainNDArray.tail.head) :: HNil)

  var lossSeq =
    for (_ <- 0 until 2000) {
      //val trainNDArray = ReadCIFAR10ToNDArray.getSGDTrainNDArray(256)
      //val loss = network.train(
      //trainNDArray.head :: makeVectorized(trainNDArray.tail.head) :: HNil)
      //println(s"loss : $loss")
      //loss
    }
}
