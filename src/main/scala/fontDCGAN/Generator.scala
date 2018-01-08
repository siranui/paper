package fontDCGAN

import breeze.linalg._
import pll._

// TODO:  引数として渡したハイパーパラメータのネットワークを生成するようにする。
case class Generator(distr: String = "He",
                     SD: Double = 0.01,
                     update_method: String = "Adam,0.5,0.999,1e-8",
                     lr: Double = 2e-4)(implicit load_path: String = "") {

  val model = new batchNet()

  // [100]
  model.add(new Affine(100, 1024, distr, SD, update_method, lr))
  model.add(new BNL(1024, 0))
  model.add(new ReLU())
  // [1024]
  model.add(new Affine(1024, 128 * 8 * 8, distr, SD, update_method, lr))
  model.add(new BNL(128 * 8 * 8, 0))
  model.add(new ReLU())
  // [8192(=128*8*8)]
  model.add(new UpSampling2D(2)((128, 8, 8)))
  model.add(new Pad(128, 1))
  // [32768(=128*16*16)]
  model.add(new i2cConv(16 + 2, 3, 64, 128, 1, distr, SD, update_method, lr))
  model.add(new BNL(64 * 16 * 16, 0))
  model.add(new ReLU())
  // [64,16,16]
  model.add(new UpSampling2D(2)((64, 16, 16)))
  model.add(new Pad(64, 1))
  // [64,32,32]
  model.add(new i2cConv(32 + 2, 3, 1, 64, 1, distr, SD, update_method, lr))
  model.add(new Tanh())
  // [1,32,32]

  if (load_path != "") {
    // model.load(load_path)
    model.load_version_iterator(load_path)
  }

}

object GeneratorTest {
  def main(args: Array[String]) {

    val gen     = Generator()
    val noize   = DenseVector.rand(100)
    val d_noize = DenseVector.rand(784)

    val res_f = gen.model.predict(Array(noize, noize))
    // res_f.foreach(v => println(v.toArray.toList))
    res_f.foreach(v => println(v.size))

    val d_res_f = gen.model.backprop(Array(d_noize, d_noize))
    d_res_f.foreach(v => println(v.size))
  }
}
