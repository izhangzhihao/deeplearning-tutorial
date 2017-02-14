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
  val NumberOfClasses: Int = 10

  //加载测试数据，我们读取100条作为测试数据
  val testNDArray =
    ReadCIFAR10ToNDArray.readFromResource(
      "/cifar-10-batches-bin/test_batch.bin",
      100)

  /**
    * 处理标签数据：将N行一列的NDArray转换为N行NumberOfClasses列的NDArray，每行对应的正确分类的值为1，其它列的值为0
    *
    * @param ndArray 标签数据
    * @return N行NumberOfClasses列的NDArray
    */
  def makeVectorized(ndArray: INDArray): INDArray = {
    val shape = ndArray.shape()

    val p = Nd4j.zeros(shape(0), NumberOfClasses)
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

  val Depth = 3

  val MiniBatchSize = 1

  val Stride = 1 // 步长

  val Padding = 1 //零填充数量

  val FilterNumber = 1 //滤波器的数量

  val FilterSize = 3 //F 滤波器的空间尺寸

  val InputSize = 3 // W 输入数据尺寸

  val PoolSize = 2 //maxPool 2*2

//  val W1 =
//    (Nd4j.randn(Array(FILTER_NUMBER, DEPTH, FILTER_SIZE, FILTER_SIZE)) / math
//      .sqrt(FILTER_SIZE / 2.0)) * 0.1

  val W1 = (1 to 27).toNDArray
    .reshape(FilterNumber, 3, 3, 3) //filter_number*depth*filter_size*filter_size

  val B1 = Nd4j.zeros(FilterNumber)

  val outputSize = (InputSize + 2 * Padding - FilterSize) / Stride + 1

  val output =
    Nd4j.zeros(MiniBatchSize, FilterNumber, outputSize, outputSize)

  val outputShape = output.shape()

  def conv_forward(
      inputData: INDArray :: INDArray :: HNil): (INDArray, INDArray) = {
    val input = inputData.head.permute(0, 1, 2, 3)

    println(input.shape().toSeq)

    assert((InputSize + 2 * Padding - FilterSize) % Stride == 0)

    assert(outputSize == InputSize)

    val inputShape = input.shape() //imageCount*3072

    val imageCount = inputShape(0) //imageCount

    val cols = Convolution.im2col(input,
                                  Array(FilterSize, FilterSize),
                                  Array(Stride, Stride),
                                  Array(0, 0))

    println("cols:" + cols)

    val cols2d = cols.reshape('c',
                              imageCount * Depth * outputSize,
                              outputSize * FilterSize * FilterSize) //115600*27

    println("cols2d" + cols2d)

    val reshapeW = W1.reshape(FilterSize * FilterSize * Depth, -1) //27*3

    //val result = Nd4j.tensorMmul(cols2d, reshapeW, Array(Array(1), Array(1))) //115600*3

    val result = cols2d.dot(reshapeW) //115600*3

    //val result = Nd4j.tensorMmul(cols2d, reshapeW, Array(Array(1), Array(0)))

    println("result" + result)

    val res = result.addRowVector(B1)

    val out = res.reshape(imageCount,
                          FilterNumber,
                          InputSize + 2 * Padding,
                          InputSize + 2 * Padding) //imageCount*3*34*34
    (out, cols)
  }

  def conv_backword(outputData: INDArray,
                    cols2d: INDArray): (INDArray, INDArray, INDArray) = {
    val db = outputData.sum(0, 2, 3)
    val dout = outputData.permute(1, 2, 3, 0).reshape(FilterNumber, -1) //3*115600

    val dw = dout
      .dot(cols2d)
      .reshape(FilterNumber, Depth, FilterSize, FilterSize)

    val dx_cols = W1.reshape(FilterNumber, -1).T.dot(dout)

    val dx = Convolution.im2col(dx_cols,
                                FilterSize,
                                FilterSize,
                                Stride,
                                Stride,
                                Padding,
                                Padding,
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

    conv_backword(out_cols._1, out_cols._2)
//    maxpool_forward()
//    maxpool_backword()
//    conv_backword()

    1.1
  }

  def softmax_loss(inputData: INDArray, label: INDArray): (Double, INDArray) =
    ???

//  val trainNDArray = ReadCIFAR10ToNDArray.readFromResource(
//    "/cifar-10-batches-bin/test_batch.bin",
//    100) //ReadCIFAR10ToNDArray.getSGDTrainNDArray(256)
//
//  val inputImageData =
//    trainNDArray.head.reshape(MINI_BATCH_SIZE, DEPTH, INPUT_SIZE, INPUT_SIZE) //imageCount*3*32*32
//
//  val padImageData = Nd4j.pad(inputImageData,
//                              Array(Array(0, 0),
//                                    Array(0, 0),
//                                    Array(PADDING, PADDING),
//                                    Array(PADDING, PADDING)),
//                              PadMode.CONSTANT)

  val x = (1 to 27).toNDArray

  private val reshapeX = x.reshape(1, 3, 3, 3)

  val padTest = Nd4j.pad(reshapeX,
                         Array(Array(0, 0),
                               Array(0, 0),
                               Array(Padding, Padding),
                               Array(Padding, Padding)),
                         PadMode.CONSTANT)

  val loss = network(padTest :: padTest :: HNil)

  var lossSeq =
    for (_ <- 0 until 2000) {
      //val trainNDArray = ReadCIFAR10ToNDArray.getSGDTrainNDArray(256)
      //val loss = network.train(
      //trainNDArray.head :: makeVectorized(trainNDArray.tail.head) :: HNil)
      //println(s"loss : $loss")
      //loss
    }
}
