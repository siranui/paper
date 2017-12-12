package hyperparameter_tune

import breeze.linalg._
import pll._
import pll.typeAlias._

// 1. 学習データ・テストデータの読み込み
// 2. ネットワークの作成
// 3. 探索するハイパーパラメータを列挙
// 4. 3から、ハイパーパラメータを選択
// 5. 選択したパラメータで学習
// 6. 4-5を繰り返す
// 7. 正解率等で評価し、評価の高くなったハイパーパラメータを出力

case class cifar10_net(distr: String = "Xavier",
                       SD: Double = 0.01,
                       update_method: String = "Adam",
                       lr: Double = 0d) {
  val model = new batchNet()
  model.add(new Affine(3 * 32 * 32, 10, distr, SD, update_method, lr))
  model.add(new SoftMax())
}

trait NB {
  def apply(distr: String, SD: Double, update_method: String, lr: Double): cifar10_net
}

object cifar10_net_builder extends NB {
  def apply(distr: String = "Xavier",
            SD: Double = 0.01,
            update_method: String = "Adam",
            lr: Double = 0d) = {
    cifar10_net(distr, SD, update_method, lr)
  }
}

object mock {
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
    val (train_err, train_acc, test_err, test_acc) = train(net, train_set, test_set)

    println(s"$train_err\t${train_acc}\t$test_err\t${test_acc}")
  }

  def train(net: cifar10_net, train_set: DATASET, test_set: DATASET) = {

    var (train_err, train_acc, test_err, test_acc) = (0d, 0d, 0d, 0d)

    // println("epoch\ttrain_err\ttrain_acc\ttest_err\ttest_acc")
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

      // println(s"$epoch\t$E\t${100d*acc.count(_==1)/acc.size}%\t$t_E\t${100d*t_acc.count(_==1)/t_acc.size}%")

      train_err = E
      train_acc = 1d * acc.count(_ == 1) / acc.size
      test_err = t_E
      test_acc = 1d * t_acc.count(_ == 1) / t_acc.size
    }

    (train_err, train_acc, test_err, test_acc)
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
