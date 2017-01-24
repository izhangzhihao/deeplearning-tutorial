# Debug

## 如何查看数据在神经网络中的某个阶段的状态

### 以softmax一节代码为例

原来的代码是这样的：
```scala
  def softmax(implicit scores: From[INDArray] ##T): To[INDArray] ##T = {
    val expScores = exp(scores)
    expScores / expScores.sum(1)
  }
```
假如要查看`exp(scores)`后的状态，需要在`exp(scores)`后添加一行代码，如下：
```scala
  def softmax(implicit scores: From[INDArray] ##T): To[INDArray] ##T = {
    val expScores: To[INDArray] ##T  = exp(scores)
      .withOutputDataHook{ data: INDArray => println(data) }
    expScores / expScores.sum(1)
  }
```

也可以省略类型，简写成这样(可能有警告，可以选择无视)：

```scala
  def softmax(implicit scores: From[INDArray] ##T): To[INDArray] ##T = {
    val expScores = exp(scores)
      .withOutputDataHook{ data => println(data) }
    expScores / expScores.sum(1)
  }
```

`withOutputDataHook`在`com.thoughtworks.deeplearning#DifferentiableAny`下，方法签名如下：

```scala
def withOutputDataHook(hook: OutputData => Unit): Layer.Aux[Input, Batch.Aux[OutputData, OutputDelta]] = ???
```

调用这个方法时可以传入一个自定义方法来进行输出或者其它操作，同时可以在这个自定义方法上打断点来查看`exp(scores)`后的状态。
假如不需要debug的时候可以注释掉新增的一行，不会影响其它地方的代码。


[完整代码](https://github.com/izhangzhihao/deeplearning-tutorial/blob/master/src/main/scala/com/thoughtworks/deeplearning/tutorial/Debug.scala)