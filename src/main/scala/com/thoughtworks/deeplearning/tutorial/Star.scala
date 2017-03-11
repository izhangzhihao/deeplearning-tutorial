package com.thoughtworks.deeplearning.tutorial
import com.thoughtworks.deeplearning.{DifferentiableINDArray, Symbolic}
import com.thoughtworks.deeplearning.DifferentiableDouble._
import com.thoughtworks.deeplearning.DifferentiableHList._
import com.thoughtworks.deeplearning.DifferentiableINDArray._
import com.thoughtworks.deeplearning.DifferentiableAny._
import com.thoughtworks.deeplearning.DifferentiableINDArray.Layers.Weight
import com.thoughtworks.deeplearning.DifferentiableINDArray.Optimizers.{
  L2Regularization,
  LearningRate,
  NesterovMomentum,
  Optimizer
}
import com.thoughtworks.deeplearning.Symbolic._
import com.thoughtworks.deeplearning.Poly.MathFunctions._
import com.thoughtworks.deeplearning.Poly._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import shapeless._

/**
  * Created by 张志豪 on 2017/2/6.
  */
object Star extends App {

  implicit val optimizerFactory = new DifferentiableINDArray.OptimizerFactory {
    override def ndArrayOptimizer(weight: Weight): Optimizer = {
      new LearningRate with L2Regularization {

        var learningRate = 0.005

        override protected def currentLearningRate(): Double = {
          learningRate *= 0.9995
          learningRate
        }

        override protected def l2Regularization: Double = 0.03
      }
    }
  }

  def fullyConnectedThenRelu(inputSize: Int, outputSize: Int)(
      implicit row: INDArray @Symbolic): INDArray @Symbolic = {
    val w =
      (Nd4j.randn(inputSize, outputSize) / math.sqrt(outputSize / 2.0)).toWeight
    val b = Nd4j.zeros(outputSize).toWeight
    max((row dot w) + b, 0.0)
  }

  def sigmoid(implicit input: INDArray @Symbolic): INDArray @Symbolic = {
    1.0 / (exp(-input) + 1.0)
  }

  def fullyConnectedThenSigmoid(inputSize: Int, outputSize: Int)(
      implicit row: INDArray @Symbolic): INDArray @Symbolic = {
    val w =
      (Nd4j.randn(inputSize, outputSize) / math.sqrt(outputSize)).toWeight
    val b = Nd4j.zeros(outputSize).toWeight
    sigmoid.compose((row dot w) + b)
  }

  def hiddenLayer(implicit input: INDArray @Symbolic): INDArray @Symbolic = {
    val layer0 = fullyConnectedThenRelu(2, 10).compose(input)
    val layer1 = fullyConnectedThenRelu(10, 10).compose(layer0)
    val layer2 = fullyConnectedThenRelu(10, 10).compose(layer1)
    fullyConnectedThenSigmoid(10, 1).compose(layer2)
  }

  val predictor = hiddenLayer

  def crossEntropy(implicit pair: (INDArray :: INDArray :: HNil) @Symbolic)
    : Double @Symbolic = {
    val score = pair.head
    val label = pair.tail.head
    -(label * log(score * 0.9 + 0.1) + (1.0 - label) * log(1.0 - score * 0.9)).sum
  }

  def network(implicit pair: (INDArray :: INDArray :: HNil) @Symbolic)
    : Double @Symbolic = {
    val input = pair.head
    val label = pair.tail.head
    val score: INDArray @Symbolic = predictor.compose(input)
    val hnilLayer: HNil @Symbolic = HNil
    crossEntropy.compose(score :: label :: hnilLayer)
  }

  val trainer = network

  val AsciiStar =
    """000000010000000
      |000000111000000
      |111111111111111
      |001111111111100
      |000011111110000
      |000111101111000
      |001110000011100
      |001000000000100""".stripMargin

  val Columns = AsciiStar.lines.next().length
  val Rows = AsciiStar.lines.length

  val coordinateData: INDArray = {
    (for {
      y <- 0 until Rows
      x <- 0 until Columns
    } yield
      Seq(x.toFloat / Rows.toFloat, y.toFloat / Columns.toFloat)).toNDArray
  }

  val labelData = (for {
    line <- AsciiStar.lines
    char <- line
  } yield Seq(Character.digit(char, 10))).toSeq.toNDArray

  for (_ <- 0 until 100) {
    var loss = 0.0
    for (_ <- 0 until 100) {
      loss = trainer.train(coordinateData :: labelData :: HNil) / Columns / Rows
    }
    val prediction: INDArray = predictor.predict(coordinateData)
    println(prediction.reshape(Rows, Columns))
    println(loss)
  }
}
