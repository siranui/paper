package DCGAN

import breeze.linalg._
import pll._

case class Discriminator() {

  val model = new batchNet()

  model.add(new i2cConv(28, 2, 64, 1, stride = 2, "He", 0.1, "Adam", 0))
  model.add(new LeakyReLU(0.2))
  model.add(new i2cConv(14, 2, 128, 64, stride = 2, "He", 0.1, "Adam", 0))
  model.add(new LeakyReLU(0.2))
  model.add(new Affine(128 * 7 * 7, 256, "He", 0.1, "Adam", 0))
  model.add(new LeakyReLU(0.2))
  model.add(new Affine(256, 1, "Xavier", 0.1, "Adam", 0))
  model.add(new Sigmoid())

}

object DiscriminatorTest {
  def main(args: Array[String]) {

    val dis     = Discriminator()
    val noize   = DenseVector.rand(784)
    val d_noize = DenseVector.rand(1)

    val res_f = dis.model.predict(Array(noize, noize))
    // res_f.foreach(v => println(v.toArray.toList))
    res_f.foreach(v => println(v.size))

    val d_res_f = dis.model.backprop(Array(d_noize, d_noize))
    d_res_f.foreach(v => println(v.size))
  }
}
