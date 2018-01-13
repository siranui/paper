package fontGLO

import pll._
// import pll.float._
import typeAlias._
import CastImplicits._
import breeze.linalg._
import scala.language.postfixOps

object zGLO2 {

  import param._

  def main(args: Array[String]) {
    val rand = new util.Random(0)

    param.setParamFromArgs(args)
    set.readConf(networkConfFile)

    val start_time   = (scala.sys.process.Process("date +%y%m%d-%H%M%S") !!).init
    val res_path     = s"${savePath}/results/${start_time}"
    val weights_path = s"${savePath}/weights/${start_time}"

    if (doSave) {
      log.info("make results and weights directory")
      val mkdir = scala.sys.process.Process(s"mkdir -p ${res_path} ${weights_path}").run
      mkdir.exitValue()
    }

    val train_d: Array[DenseVector[T]] = dataSources match {
      case Nil =>
        utils.read(dataSource, data_size)
      case _ =>
        val tmp = dataSources.map(d => utils.read(d)).reduce(_ ++ _)
        data_size = tmp.size
        tmp
    }

    log.debug(s"data size is $data_size")

    var Z = Array.ofDim[DenseVector[T]](data_size)
    Z = Z.map { _ =>
      DenseVector.fill[T](zdim) {
        rand.nextGaussian / math.sqrt(data_size)
      }
    }
    if (LOAD_Z != "") Z = utils.read(LOAD_Z)

    val g = Generator(distr, SD, update_method, lr, zdim)(LOAD_PARAM_G)
    g.model.layers.foreach(println)

    val d_alpha: T = 0.01

    log.info("********** training start **********")
    // training
    for (e <- 0 until epoch) {
      var E: T = 0
      var unusedIdx =
        if (doShuffle) rand.shuffle(List.range(0, data_size))
        else List.range(0, data_size)

      while (unusedIdx.nonEmpty) {
        val batchMask = unusedIdx.take(batch)
        unusedIdx = unusedIdx.drop(batch)

        val xs = batchMask.map(idx => Z(idx)).toArray
        val ts = batchMask.map(idx => train_d(idx)).toArray
        val ys = g.model.predict(xs)

        var d = Array[DenseVector[T]]()
        Loss match {
          case "Laplacian" | "laplacian" | "Lap" | "lap" =>
            E += err.calc_Lap1_loss(ys, ts)
            d = grad.calc_Lap1_grad(ys, ts)
          case "L2" | _ =>
            E += err.calc_L2(ys, ts)
            d = grad.calc_L2_grad(ys, ts)
        }

        // g.model.update(d)
        val z_grad = g.model.backprop(d)
        g.model.update()
        g.model.reset()

        (0 until batch).foreach { idx =>
          Z(batchMask(idx)) -= d_alpha * z_grad(idx)
        }
      }

      // save
      val saveCondition: Boolean = (e == 0) || (e % (epoch / saveTime) == 0) || (e == epoch - 1)
      if (doSave && saveCondition) {
        // val filename = s"batch_font_GLO_ds${data_size}_epoch${e}of${epoch}_batch${batch}_pad${pad}_stride${stride}.txt"
        val filename =
          f"batch_font_GLO_ds${data_size}_epoch${e}%05dof${epoch}%05d_batch${batch}.txt"

        var ys = List[DenseVector[Int]]()
        for (z <- Z) {
          val y = g.model.predict(z).map(o => (o * 256).toInt)
          ys = y :: ys
          g.model.reset()
        }

        // save generated image
        utils.write(s"${res_path}/${filename}", ys.reverse)
        log.info("********** save generated image **********")

        // save weights
        g.model.save_one_file(s"${weights_path}/e${e}_Gen.weight")
        log.info("********** save weights **********")

        // save learning Z
        utils.write(s"${weights_path}/e${e}_Z.txt", Z)
        log.info("********** save learning Z **********")
      }

      // output Error
      println(s"$e, $E")
    }
    log.info("********** training finish **********")

    println(args.toList)
  }

}