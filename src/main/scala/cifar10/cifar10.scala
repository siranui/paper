package cifar10

import breeze.linalg._
import pll._

case class cifar10_net() {
  val model = new batchNet()
  model.add(new i2cConv(32, 5, 10, 3, 1, "He", 0.01, "Adam", 0d))
  model.add(new Pooling(2, 2)(10, 28, 28))
  model.add(new LeakyReLU())
  // [10,14,14]
  model.add(new i2cConv(14, 5, 10, 10, 1, "He", 0.01, "Adam", 0d))
  model.add(new Pooling(2, 2)(10, 10, 10))
  model.add(new LeakyReLU())
  // [10,5,5]
  model.add(new i2cConv(5, 3, 10, 10, 1, "He", 0.01, "Adam", 0d))
  model.add(new LeakyReLU())
  model.add(new Affine(3 * 3 * 10, 100, "He", 0.01, "Adam", 0d))
  model.add(new ReLU())
  model.add(new Affine(100, 10, "Xavier", 0.01, "Adam", 0d))
  model.add(new SoftMax())
}

object cifar10 {
  import pll.typeAlias._
  // // types
  // type T       = Double
  // type DV      = DenseVector[T]
  // type DM      = DenseMatrix[T]
  // type DATASET = Array[(DV, DV)]

  // params
  var train_size = 50000
  var test_size  = 10000

  val DATA_DIR = sys.env.getOrElse("XDG_DATA_HOME", "/home/share") + "/cifar10"

  var TRAIN_DATA_PATH = s"$DATA_DIR/train-d.txt"
  var TRAIN_TAG_PATH  = s"$DATA_DIR/train-t.txt"
  var TEST_DATA_PATH  = s"$DATA_DIR/test-d.txt"
  var TEST_TAG_PATH   = s"$DATA_DIR/test-t.txt"

  var do_reshape = true
  var NUM_EPOCH  = 20
  var BATCH_SIZE = 32

  val rand = new util.Random(0)

  def main(args: Array[String]) {
    args_process(args)

    println("--- data loading ---")
    val Seq((train_data, train_tag), (test_data, test_tag)) = data_load()

    val train_set: DATASET = train_data zip train_tag
    val test_set: DATASET  = test_data zip test_tag

    println("--- network initialize ---")
    val net = cifar10_net()

    println("--- training start ---")

    println("epoch\ttrain_err\ttrain_acc\ttest_err\ttest_acc")
    for (epoch <- 0 until NUM_EPOCH) {
      // train
      val train_batch = rand.shuffle(train_set.iterator).toArray.take(BATCH_SIZE)
      val ys          = net.model.predict(train_batch.map(_._1))

      val E = err.calc_cross_entropy_loss(ys, train_batch.map(_._2))
      val acc = (ys zip train_batch.map(_._2)).map {
        case (y, t) =>
          if (argmax(y) == argmax(t)) 1d
          else 0d
      }

      val d = grad.calc_cross_entropy_grad(ys, train_batch.map(_._2))
      net.model.update(d)

      // test
      val test_batch = rand.shuffle(test_set.iterator).toArray.take(BATCH_SIZE)
      val t_ys       = net.model.predict(test_batch.map(_._1))
      net.model.reset()

      val t_E = err.calc_cross_entropy_loss(t_ys, test_batch.map(_._2))
      val t_acc = (t_ys zip test_batch.map(_._2)).map {
        case (y, t) =>
          if (argmax(y) == argmax(t)) 1
          else 0
      }

      println(
        s"$epoch\t$E\t${100d * acc.count(_ == 1) / acc.size}%\t$t_E\t${100d * t_acc.count(_ == 1) / t_acc.size}%")
    }

  }

  def args_process(args: Array[String]): Unit = {
    var i = 0
    while (i < args.size) {
      args(i) match {
        case "--train-size" =>
          train_size = args(i + 1).toInt
          i += 2
        case "--test-size" =>
          test_size = args(i + 1).toInt
          i += 2
        case "--train-data-path" =>
          TRAIN_DATA_PATH = args(i + 1)
          i += 2
        case "--test-data-path" =>
          TEST_DATA_PATH = args(i + 1)
          i += 2
        case "--train-tag-path" =>
          TRAIN_TAG_PATH = args(i + 1)
          i += 2
        case "--test-tag-path" =>
          TEST_TAG_PATH = args(i + 1)
          i += 2
        case "--do-reshape" =>
          do_reshape = args(i + 1).toBoolean
          i += 2
        case "--epoch" =>
          NUM_EPOCH = args(i + 1).toInt
          i += 2
        case "--batch-size" =>
          BATCH_SIZE = args(i + 1).toInt
          i += 2
        case other =>
          println(s"$other is not exist.")
          sys.exit()
      }
    }
  }

  def reshape(src: Array[Array[Double]]) = {
    val r = src.map { c =>
      (0 until c.size by 3).map(k => c(k)).toArray
    }
    val g = src.map { c =>
      (1 until c.size by 3).map(k => c(k)).toArray
    }
    val b = src.map { c =>
      (2 until c.size by 3).map(k => c(k)).toArray
    }
    val dst: Array[DenseVector[Double]] = (for (idx <- 0 until r.size) yield {
      DenseVector(r(idx) ++ g(idx) ++ b(idx))
    }).toArray
    dst
  }

  def data_load(): Seq[(Array[DenseVector[Double]], Array[DenseVector[Double]])] = {

    val tr_d: Array[Array[Double]] = io.Source
      .fromFile(TRAIN_DATA_PATH)
      .getLines
      .take(train_size)
      .map(_.split(",").map(_.toDouble))
      .toArray
    val te_d: Array[Array[Double]] = io.Source
      .fromFile(TEST_DATA_PATH)
      .getLines
      .take(test_size)
      .map(_.split(",").map(_.toDouble))
      .toArray
    val tr_t: Array[Int] = io.Source
      .fromFile(TRAIN_TAG_PATH)
      .getLines
      .map(_.split(",").map(_.toInt))
      .toArray
      .flatten
      .take(train_size)
    val te_t: Array[Int] = io.Source
      .fromFile(TEST_TAG_PATH)
      .getLines
      .map(_.split(",").map(_.toInt))
      .toArray
      .flatten
      .take(test_size)

    val train_data: Array[DenseVector[Double]] = do_reshape match {
      case true => reshape(tr_d)
      case false =>
        tr_d.map { ad =>
          DenseVector(ad)
        }
    }
    val test_data: Array[DenseVector[Double]] = do_reshape match {
      case true => reshape(te_d)
      case false =>
        te_d.map { ad =>
          DenseVector(ad)
        }
    }
    val train_tag: Array[DenseVector[Double]] = tr_t.map { t =>
      utils.oneHot(t)
    }
    val test_tag: Array[DenseVector[Double]] = te_t.map { t =>
      utils.oneHot(t)
    }

    Seq((train_data, train_tag), (test_data, test_tag))

  }

}
