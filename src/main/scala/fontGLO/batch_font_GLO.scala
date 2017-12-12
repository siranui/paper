package fontGLO

import pll._
import breeze.linalg._
import scala.language.postfixOps

class batch_font_GLO(var LapDir: Int = 8) extends batchNet {

  val Lap = Convolution(32, 3, 1, 1, 1, "", 1d, "", 1d)

  val l: DenseVector[Double] = LapDir match {
    case 4 =>
      DenseVector[Double](
        0, 1, 0, 1, -4, 1, 0, 1, 0
      )
    case 8 | _ =>
      DenseVector[Double](
        1, 1, 1, 1, -8, 1, 1, 1, 1
      )
  }
  Lap.F = Array(Array(l))

  def calc_Lap_loss(y: DenseVector[Double], t: DenseVector[Double]): Double = {
    math.pow(2d, -2d) * sum((Lap.forward(y) - Lap.forward(t)).map(math.abs))
  }

  def calc_Lap_grad(y: DenseVector[Double], t: DenseVector[Double]): DenseVector[Double] = {
    val o = Lap.forward(y) - Lap.forward(t)
    val W = Lap.filter2Weight(l, 32 * 32, 1)
    W.t * o.map(a => if (a > 0) 1d else -1d)
  }

  def calc_Lap_loss(ys: Array[DenseVector[Double]], ts: Array[DenseVector[Double]]): Double = {
    var E = 0d
    for ((y, t) <- ys zip ts) {
      E += calc_Lap_loss(y, t)
    }
    E
  }

  def calc_Lap_grad(
      ys: Array[DenseVector[Double]],
      ts: Array[DenseVector[Double]]
  ): Array[DenseVector[Double]] = {

    val grads = for ((y, t) <- ys zip ts) yield {
      calc_Lap_grad(y, t)
    }
    grads

  }

}

object batch_font_GLO {

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

    val g: batch_font_GLO = set.connectNetwork(new batch_font_GLO(LapDir))
    g.layers.foreach(println)

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
        val ys = g.predict(xs)

        var d = Array[DenseVector[Double]]()
        Loss match {
          case "Laplacian" | "laplacian" | "Lap" | "lap" =>
            E += g.calc_Lap_loss(ys, ts)
            d = g.calc_Lap_grad(ys, ts)
          case "L2" | _ =>
            E += err.calc_L2(ys, ts)
            d = grad.calc_L2_grad(ys, ts)
        }

        g.update(d)
      }

      // save
      val saveCondition: Boolean = (e == 0) || (e % (epoch / saveTime) == 0) || (e == epoch - 1)
      if (doSave && saveCondition) {
        // val filename = s"batch_font_GLO_ds${data_size}_epoch${e}of${epoch}_batch${batch}_pad${pad}_stride${stride}.txt"
        val filename = s"batch_font_GLO_ds${data_size}_epoch${e}of${epoch}_batch${batch}.txt"

        var ys = List[DenseVector[Int]]()
        for (z <- Z) {
          val y = g.predict(z).map(o => (o * 256).toInt)
          ys = y :: ys
          g.reset()
        }

        utils.write(s"${res_path}/${filename}", ys.reverse)

        // save weights
        for (i <- g.layers.indices) {
          val LAYER = g.layers(i).getClass.toString.split(" ").last.drop(4)
          // MEMO:
          //   (g.layers(i).getClass()).toString ==> 'class pll.hogehoge'
          //   (g.layers(i).getClass()).toString.split(" ").last.drop(4) ==> 'hogehoge'

          g.layers(i).save(s"${weights_path}/${LAYER}_${i}_${filename}")
        }
      }

      // output Error
      println(s"$e, $E")
    }

    println(args.toList)
  }

}
