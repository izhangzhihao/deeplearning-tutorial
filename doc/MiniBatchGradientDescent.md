# Mini-Batch Gradient Descent

## 背景

通过使用小批量数据随机梯度下降快速实现神经网络参数更新。

## 准备工作

前四步与SoftmaxLinearClassifier中相同，不再赘述。

5.在类ReadCIFAR10ToNDArray中增加若干方法和属性,代码如下：

```scala
object ReadCIFAR10ToNDArray {

  /**
    * 目的是将存储数据的文件读取到内存中，以后就从这里拿数据，避免了频繁的读取文件。
    *
    */
  lazy val fileBytesSeq: IndexedSeq[Array[Byte]] = {
    for {
      fileIndex <- 1 to 5
      //if you are using IDE
      //inputStream = getClass.getResourceAsStream("/cifar-10-batches-bin/data_batch_" + fileIndex + ".bin")
    
      //if you are using jupyter notebook,please use this
      inputStream = new FileInputStream(sys.env("PWD") + "/src/main/resources" + "/cifar-10-batches-bin/data_batch_" + fileIndex + ".bin")
    } yield readFromInputStream(inputStream)
  }

  /**
    * 从inputStream中读取byte
    *
    * @param inputStream
    * @return
    */
  def readFromInputStream(inputStream: InputStream): Array[Byte] = {
    try {
      val bytes = Array.range(0, 3073 * 10000).map(_.toByte)
      inputStream.read(bytes)
      bytes
    } finally {
      inputStream.close()
    }
  }

  /**
    * 随机获取count个train数据
    *
    * @return
    */
  def getSGDTrainNDArray(count: Int): INDArray :: INDArray :: HNil = {
    //生成0到4的随机数
    val randomIndex = (new util.Random).nextInt(5)
    val bytes = fileBytesSeq(randomIndex)

    val indexList = randomList(10000, count)

    val labels: Seq[Double] = for (index <- 0 until count)
      yield bytes(indexList(index) * 3073).toDouble

    val pixels: Seq[Seq[Double]] = for (index <- 0 until count)
      yield {
        for (pixelItem <- 1 until 3073)
          yield normalizePixel(bytes(indexList(index) * 3073 + pixelItem).toDouble)
      }

    val labelsNDArray = labels.toNDArray.reshape(count, 1)
    val pixelsNDArray = pixels.toNDArray

    pixelsNDArray :: labelsNDArray :: HNil

  }

  /**
    * 从固定范围内获取count个数字组成的集合
    *
    * @param arrange 范围
    * @param count   个数
    * @return
    */
  def randomList(arrange: Int, count: Int): List[Int] = {
    var resultList: List[Int] = Nil
    while (resultList.length < count) {
      val randomNum = (new util.Random).nextInt(arrange)
      if (!resultList.contains(randomNum)) {
        resultList = resultList ::: List(randomNum)
      }
    }
    resultList
  }
}
```
6.[Mini-Batch Gradient Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent): 在大规模数据训练时，数据可以达到百万级量级。如果计算整个训练集，来获得仅仅一个参数的更新速度就太慢了。一个常用的方法是计算训练集中的小批量（batches）数据以提升参数更新速度。

7.如果你使用IntelliJ或者eclipse等其它IDE，智能提示可能会失效，代码有部分可能会爆红，这是IDE的问题，代码本身并无问题。
   

## 构建神经网络

1.新建一个Scala类MiniBatchGradientDescent

2.从CIFAR10 database中读取测试数据的图片和标签信息，注意：这里和与SoftmaxLinearClassifier中不同

```scala
  //CIFAR10中的图片共有10个分类(airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck)
  val CLASSES: Int = 10

  //加载测试数据，我们读取100条作为测试数据
  val testNDArray =
    ReadCIFAR10ToNDArray.readFromResource("/cifar-10-batches-bin/test_batch.bin", 100)
```

3.编写处理标签数据的工具方法，将N行一列的NDArray转换为N行CLASSES列的NDArray，每行对应的正确分类的值为1，其它列的值为0。这样做是为了向cross-entropy loss公式靠拢


```scala
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
```

4.分离和处理图像和标签数据，注意：这里和与SoftmaxLinearClassifier中不同

```scala
  val test_data = testNDArray.head

  val test_expect_result = testNDArray.tail.head
  
  val test_p = makeVectorized(test_expect_result)
```

5.编写softmax函数,和准备一节中的softmax公式对应

```scala
  def softmax(implicit scores: From[INDArray] ##T): To[INDArray] ##T = {
    val expScores = exp(scores)
    expScores / expScores.sum(1)
  }
```

6.跟定义一个方法一样定义一个神经网络并初始化Weight，Weight应该是一个N*CLASSES的INDArray,每个图片对应每个分类都有一个评分。[什么是Weight](https://github.com/ThoughtWorksInc/DeepLearning.scala/wiki/Getting-Started#231--weight-intialization)

```scala
  def createMyNeuralNetwork(implicit input: From[INDArray] ##T): To[INDArray] ##T = {
    val initialValueOfWeight = Nd4j.randn(3072, CLASSES) * 0.001
    val weight: To[INDArray] ##T = initialValueOfWeight.toWeight
    val result: To[INDArray] ##T = input dot weight
    softmax.compose(result) //对结果调用softmax方法，压缩结果值在0到1之间方便处理
  }
  val myNeuralNetwork: FromTo[INDArray, INDArray] ##T = createMyNeuralNetwork
```

7.设置学习率，学习率是Weight变化的快慢的直观描述，学习率设置的过小会导致loss下降的很慢，需要更长时间来训练，学习率设置的过大虽然刚开始下降很快但是会导致在接近最低点的时候在附近徘徊loss下降会非常慢。

```scala
  implicit def optimizer: Optimizer = new LearningRate {
    def currentLearningRate() = 0.00001
  }
```

8.编写损失函数Loss Function，将此次判断的结果和真实结果进行计算得出cross-entropy loss并返回

```scala
  def lossFunction(implicit pair: From[INDArray :: INDArray :: HNil] ##T): To[Double] ##T = {
    val input = pair.head
    val expectedOutput = pair.tail.head
    val probabilities = myNeuralNetwork.compose(input)

    -(expectedOutput * log(probabilities)).sum //此处和准备一节中的交叉熵损失对应
  }
```

9.训练神经网络并观察每次训练loss的变化，loss的变化趋势是降低，但是不是每次都降低(前途是光明的，道路是曲折的)

```scala
  for (_ <- 0 until 2000) {
    val trainNDArray = ReadCIFAR10ToNDArray.getSGDTrainNDArray(256)
    val loss = lossFunction.train(trainNDArray.head :: makeVectorized(trainNDArray.tail.head) :: HNil)
    println(s"loss : $loss")
  }
```

10.使用训练后的神经网络判断测试数据的标签

```scala
  val result = myNeuralNetwork.predict(test_data)
  println(s"result: $result") //输出判断结果
```

11.编写工具方法，从一行INDArray中获得值最大的元素所在的列，目的是获得神经网络判断的结果，方便和原始标签比较以得出正确率。

```scala
  /**
    * 从一行INDArray中获得值最大的元素所在的列
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
```

12.判断神经网络对测试数据分类判断的正确率，正确率应该在40%左右。

```scala
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
```

13.[完整代码](https://github.com/izhangzhihao/deeplearning-tutorial/blob/master/src/main/scala/com/thoughtworks/deeplearning/tutorial/MiniBatchGradientDescent.scala)