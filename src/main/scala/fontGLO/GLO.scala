package fontGLO

import pll._
import breeze.linalg._
import scala.language.postfixOps

object GLO {

  import param._

  def main(args: Array[String]) {
    val rand = new util.Random(0)

    param.setParamFromArgs(args)
    set.readConf(networkConfFile)

    val start_time   = (scala.sys.process.Process("date +%y%m%d-%H%M%S") !!).init
    val res_path     = s"${savePath}/results/${start_time}"
    val weights_path = s"${savePath}/weights/${start_time}"

    if (doSave) {
      val mkdir = scala.sys.process.Process(s"mkdir -p ${res_path} ${weights_path}").run
      mkdir.exitValue()
    }

    val train_d = utils.read(dataSource, data_size)

    var Z = Array.ofDim[DenseVector[Double]](data_size)
    Z = Z.map { _ =>
      DenseVector.fill(zdim) {
        rand.nextGaussian / math.sqrt(data_size)
      }
    }

    val g = fontDCGAN.Generator(distr, SD, update_method, lr)
    g.model.layers.foreach(println)

    // training
    for (e <- 0 until epoch) {
      var E = 0d
      var unusedIdx =
        if (doShuffle) rand.shuffle(List.range(0, data_size))
        else List.range(0, data_size)

      while (unusedIdx.nonEmpty) {
        val batchMask = unusedIdx.take(batch)
        unusedIdx = unusedIdx.drop(batch)

        val xs = batchMask.map(idx => Z(idx)).toArray
        val ts = batchMask.map(idx => train_d(idx)).toArray
        val ys = g.model.predict(xs)

        var d = Array[DenseVector[Double]]()
        Loss match {
          case "Laplacian" | "laplacian" | "Lap" | "lap" =>
            E += err.calc_Lap1_loss(ys, ts)
            d = grad.calc_Lap1_grad(ys, ts)
          case "L2" | _ =>
            E += err.calc_L2(ys, ts)
            d = grad.calc_L2_grad(ys, ts)
        }

        g.model.update(d)
      }

      // save
      val saveCondition: Boolean = (e == 0) || (e % (epoch / saveTime) == 0) || (e == epoch - 1)
      if (doSave && saveCondition) {
        // val filename = s"batch_font_GLO_ds${data_size}_epoch${e}of${epoch}_batch${batch}_pad${pad}_stride${stride}.txt"
        val filename = s"batch_font_GLO_ds${data_size}_epoch${e}of${epoch}_batch${batch}.txt"

        var ys = List[DenseVector[Int]]()
        for (z <- Z) {
          val y = g.model.predict(z).map(o => (o * 256).toInt)
          ys = y :: ys
          g.model.reset()
        }

        utils.write(s"${res_path}/${filename}", ys.reverse)

        // save weights
        g.model.save_one_file(s"${weights_path}/${filename}_Gen_epoch${e}.weight")
      }

      // output Error
      println(s"$e, $E")
    }

    println(args.toList)
  }

}
