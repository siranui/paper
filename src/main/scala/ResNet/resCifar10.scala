package ResNet

import pll._
import breeze.linalg._

object resCifar10 {
  // read cifar10's data.
  val ds = 1000
  val ts = 1000

  val train_d: Array[Array[Double]] = io.Source.fromFile("/home/share/cifar10/train-d.txt").getLines.take(ds).map(_.split(",").map(_.toDouble / 256d)).toArray
  val r = train_d.map{ c => (0 until c.size by 3).map(k => c(k)).toArray }
  val g = train_d.map{ c => (1 until c.size by 3).map(k => c(k)).toArray }
  val b = train_d.map{ c => (2 until c.size by 3).map(k => c(k)).toArray }
  val data: List[DenseVector[Double]] = (for (idx <- 0 until r.size) yield {
    DenseVector(r(idx) ++ g(idx) ++ b(idx))
  }).toList
  val tag: Array[Int] = io.Source.fromFile("/home/share/cifar10/train-t.txt").getLines.map(_.split(",").take(ds).map(_.toInt)).toArray.flatten



  // read test data.
  val test_d: Array[Array[Double]] = io.Source.fromFile("/home/share/cifar10/test-d.txt").getLines.take(ds).map(_.split(",").map(_.toDouble / 256d)).toArray
  val tr = test_d.map{ c => (0 until c.size by 3).map(k => c(k)).toArray }
  val tg = test_d.map{ c => (1 until c.size by 3).map(k => c(k)).toArray }
  val tb = test_d.map{ c => (2 until c.size by 3).map(k => c(k)).toArray }
  val t_data: List[DenseVector[Double]] = (for (idx <- 0 until r.size) yield {
    DenseVector(tr(idx) ++ tg(idx) ++ tb(idx))
  }).toList
  val t_tag: Array[Int] = io.Source.fromFile("/home/share/cifar10/test-t.txt").getLines.map(_.split(",").take(ds).map(_.toInt)).toArray.flatten



  val epoch = 100

  def main(args: Array[String]) {

    // create network
    val net = new Network()
    net.add(
      new ResNet(
        Seq(
          new Pad(channel = 3, width = 1),
          new i2cConv(input_width = 32 + 2, filter_width = 3, filter_set = 3, channel = 3, stride = 1, distr = "Gaussian", SD = 0.1, update_method = "SGD", lr = 0.01),
          new ReLU(),
          new Pad(channel = 3, width = 1),
          new i2cConv(input_width = 32 + 2, filter_width = 3, filter_set = 3, channel = 3, stride = 1, distr = "Gaussian", SD = 0.1, update_method = "SGD", lr = 0.01))))
    net.add(new ReLU())
    net.add(new i2cConv(input_width = 32, filter_width = 2, filter_set = 4, channel = 3, stride = 2, distr = "Gaussian", SD = 0.1, update_method = "SGD", lr = 0.01))
    net.add(
      new ResNet(
        Seq(
          new Pad(channel = 4, width = 1),
          new i2cConv(input_width = 16 + 2, filter_width = 3, filter_set = 4, channel = 4, stride = 1, distr = "Gaussian", SD = 0.1, update_method = "SGD", lr = 0.01),
          new ReLU(),
          new Pad(channel = 4, width = 1),
          new i2cConv(input_width = 16 + 2, filter_width = 3, filter_set = 4, channel = 4, stride = 1, distr = "Gaussian", SD = 0.1, update_method = "SGD", lr = 0.01))))
    net.add(new ReLU())
    net.add(new Affine(16 * 16 * 4, 10, "Xavier", 0.1, "SGD", 0.01))
    net.add(new SoftMax())

    val confusionMatrixFile = new java.io.File("confusion_matrix.csv")
    val test_confusionMatrixFile = new java.io.File("test_confusion_matrix.csv")

    // training start
    for (e <- 0 until epoch) {

      val dataset = data zip tag
      val testset = t_data zip t_tag

      var E = 0d
      var tE = 0d

      var acc = 0d
      var tacc = 0d

      val mixMat = DenseMatrix.zeros[Int](10, 10)
      val tmixMat = DenseMatrix.zeros[Int](10, 10)

      // training
      for ((data, tag) <- dataset) {
        val y = net.predict(data)
        val t = utils.oneHot(tag.toInt)

        if (tag == argmax(y)) acc += 1

        mixMat(tag.toInt, argmax(y).toInt) += 1

        E += err.calc_cross_entropy_loss(y, t)

        val d = grad.calc_L2_grad(y, t)

        net.update(d)
      }

      //test
      for ((data, tag) <- testset) {
        val y = net.predict(data)
        val t = utils.oneHot(tag.toInt)

        if (tag == argmax(y)) tacc += 1

        tmixMat(tag.toInt, argmax(y).toInt) += 1

        tE += err.calc_cross_entropy_loss(y, t)
      }

      //logging
      println(s"$e, E:$E, acc:${acc / ds * 100}%, tE:$tE, tacc:${tacc / ts * 100}%")
      println("mixmat(tag, argmax(y))")
      println("--train--")
      println(mixMat)
      println("--test--")
      println(tmixMat)
      println()
      // println("mixmat(tag, argmax(y))")
      // println(s"train\n$mixMat")

      csvwrite(confusionMatrixFile,convert(mixMat,Double))
      csvwrite(test_confusionMatrixFile,convert(tmixMat,Double))
    }

  }
}
