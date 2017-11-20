package DCGAN

import breeze.linalg._
import pll._

case class Generator() {

  val model = new batchNet()

  model.add(new Affine(100, 1024, "Xavier", 0.1, "Adam", 0))
  model.add(new BNL(1024, 0))
  model.add(new ReLU())
  model.add(new Affine(1024, 128 * 7 * 7, "Xavier", 0.1, "Adam", 0))
  model.add(new BNL(128 * 7 * 7, 0))
  model.add(new ReLU())
  model.add(new UpSampling2D(2)((128, 7, 7)))
  model.add(new Pad(128, 1))
  model.add(new i2cConv(14 + 2, 3, 64, 128))
  model.add(new BNL(64 * 14 * 14, 0))
  model.add(new ReLU())
  model.add(new UpSampling2D(2)((64, 14, 14)))
  model.add(new Pad(64, 1))
  model.add(new i2cConv(28 + 2, 3, 1, 64))
  model.add(new Tanh())

}

object GeneratorTest {
  def main(args: Array[String]) {

    val gen = Generator()
    val noize = DenseVector.rand(100)
    val d_noize = DenseVector.rand(784)

    val res_f = gen.model.predict(Array(noize, noize))
    // res_f.foreach(v => println(v.toArray.toList))
    res_f.foreach(v => println(v.size))

    val d_res_f = gen.model.backprop(Array(d_noize, d_noize))
    d_res_f.foreach(v => println(v.size))
  }
}
