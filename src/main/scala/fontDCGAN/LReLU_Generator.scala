package fontDCGAN

import breeze.linalg._
import pll._

// TODO:  引数として渡したハイパーパラメータのネットワークを生成するようにする。
case class LReLUGenerator(distr: String = "He",
                          SD: Double = 0.01,
                          update_method: String = "Adam,0.5,0.999,1e-8",
                          lr: Double = 2e-4)(implicit load_path: String = "") {

  val model = new batchNet()

  // [100]
  model.add(new Affine(100, 1024, distr, SD, update_method, lr))
  model.add(new BNL(1024, 0))
  model.add(new LeakyReLU())
  // [1024]
  model.add(new Affine(1024, 128 * 8 * 8, distr, SD, update_method, lr))
  model.add(new BNL(128 * 8 * 8, 0))
  model.add(new LeakyReLU())
  // [8192(=128*8*8)]
  model.add(new UpSampling2D(2)((128, 8, 8)))
  model.add(new Pad(128, 1))
  // [32768(=128*16*16)]
  model.add(new i2cConv(16 + 2, 3, 64, 128, 1, distr, SD, update_method, lr))
  model.add(new BNL(64 * 16 * 16, 0))
  model.add(new LeakyReLU())
  // [64,16,16]
  model.add(new UpSampling2D(2)((64, 16, 16)))
  model.add(new Pad(64, 1))
  // [64,32,32]
  model.add(new i2cConv(32 + 2, 3, 1, 64, 1, distr, SD, update_method, lr))
  model.add(new Tanh())
  // [1,32,32]

  if (load_path != "") {
    model.load(load_path)
  }

}
