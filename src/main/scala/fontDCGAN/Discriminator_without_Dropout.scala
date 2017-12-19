package fontDCGAN

import breeze.linalg._
import pll._

// TODO:  引数として渡したハイパーパラメータのネットワークを生成するようにする。
case class Discriminator_without_Dropout(distr: String = "He",
                                         SD: Double = 0.01,
                                         update_method: String = "Adam,0.1,0.999,1e-8",
                                         lr: Double = 1e-5)(implicit load_path: String = "") {

  val model = new batchNet()

  // [1,32,32]
  model.add(new i2cConv(32, 2, 64, 1, stride = 2, distr, SD, update_method, lr))
  model.add(new LeakyReLU(0.2))
  // [64,16,16]
  model.add(new i2cConv(16, 2, 128, 64, stride = 2, distr, SD, update_method, lr))
  model.add(new LeakyReLU(0.2))
  // [128,8,8]
  model.add(new Affine(128 * 8 * 8, 256, distr, SD, update_method, lr))
  model.add(new LeakyReLU(0.2))
  // [256]
  // model.add(new Dropout(0.5) )
  model.add(new Affine(256, 1, distr, SD, update_method, lr))
  model.add(new Sigmoid())
  // [1]

  if (load_path != "") {
    model.load(load_path)
  }
}
