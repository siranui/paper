package fontGLO


import pll._
import breeze.linalg._
import scala.language.postfixOps

object Up_convAE {

  import param._
  import pll.actors._
  import pll.actors.Monitor._

  val rand = new util.Random(0)

  def main(args: Array[String]) {

    param.setParamFromArgs(args)
    set.readConf(networkConfFile)

    val start_time = (scala.sys.process.Process("date +%y%m%d-%H%M%S") !!).init
    val res_path = s"${savePath}/results/${start_time}"
    val weights_path = s"${savePath}/weights/${start_time}"

    if (doSave) {
      val mkdir = scala.sys.process.Process(s"mkdir -p ${res_path} ${weights_path}").run
      mkdir.exitValue()
    }

    val data = utils.read(dataSource, data_size * 2)
    val train_d = data.take(data_size)
    val test_d = data.drop(data_size)

    var Z = Array.ofDim[DenseVector[Double]](data_size)
    Z = Z.map { _ =>
      DenseVector.fill(32*32) {
        rand.nextGaussian / math.sqrt(data_size)
      }
    }


    val g = new batchNet()
    // [1,32,32] -> [3,16,16]
    g.add(new Pad(1, 1))
      .add(new i2cConv(32+2, 4, 3, 1, 2, "He", 0.01, "RMSProp", 0.01))
      .add(new LeakyReLU())
    // [3,16,16] -> [5,8,8]
    g.add(new Pad(3, 1))
      .add(new i2cConv(16+2, 4, 5, 3, 2, "He", 0.01, "RMSProp", 0.01))
      .add(new LeakyReLU())
    // [5,8,8] -> [10,4,4]
    g.add(new Pad(5, 1))
      .add(new i2cConv(8+2, 4, 10, 5, 2, "He", 0.01, "RMSProp", 0.01))
      .add(new LeakyReLU())
    // [10,4,4] -> [5,8,8]
    g.add(new UpSampling2D(2)((10,4,4)))
      .add(new Pad(10,1))
      .add(new i2cConv(8+2, 3, 5, 10, 1, "He", 0.01, "RMSProp", 0.01))
      .add(new LeakyReLU())
    // [5,8,8] -> [3,16,16]
    g.add(new UpSampling2D(2)((5,8,8)))
      .add(new Pad(5, 1))
      .add(new i2cConv(16+2, 3, 3, 5, 1, "He", 0.01, "RMSProp", 0.01))
      .add(new LeakyReLU())
    // [3,16,16] -> [1,32,32]
    g.add(new UpSampling2D(2)((3,16,16)))
      .add(new Pad(3, 1))
      .add(new i2cConv(32+2, 3, 1, 3, 1, "He", 0.01, "RMSProp", 0.01))
      .add(new Tanh())

    // for monitoring.
    val system = akka.actor.ActorSystem("system")
    val err_fig = breeze.plot.Figure("error-figure")
    val img_fig = breeze.plot.Figure("train-image-figure")
    val test_img_fig = breeze.plot.Figure("test-image-figure")
    val err_actor = system.actorOf(Monitor.props(err_fig), "error-actor")
    val img_actor = system.actorOf(Monitor.props(img_fig), "train-image-actor")
    val test_img_actor = system.actorOf(Monitor.props(test_img_fig), "test_image-actor")
    var E_list:  List[Double] = Nil
    var tE_list: List[Double] = Nil

    // training
    for (e <- 0 until epoch) {

      val train = g.batch_train(train_d, train_d, batch, err.calc_L2, grad.calc_L2_grad)
      val test = g.test(test_d, test_d, err.calc_L2)

      println(s"$e, E: ${train._1}, tE: ${test._1}")
      println(train._2.flatten.head)


      if(display){
        E_list  = train._1 :: E_list
        tE_list = test._1  :: tE_list

        // plotting change of error.
        // graph.Line(err_fig, Seq(E_list, tE_list).map(l => DenseVector(l.reverse.toArray)),epoch,2)
        err_actor ! Line(xs = Seq(E_list, tE_list).map(l => DenseVector(l.reverse.toArray)), x_max = epoch, row = 2)
        // visualize generated image
        // val tr = train._2.flatten.take(50).map{ m =>
        //   reshape(m, 32, 32).t * 256d
        // }
        img_actor ! Image_v(xs = train._2.flatten.take(50), row = 5)
        test_img_actor ! Image_v(xs = test._2.take(50), row = 5)
      }


      // save
      val saveCondition: Boolean = (e == 0) || (e % (epoch / saveTime) == 0) || (e == epoch - 1)
      if (doSave && saveCondition) {
        val filename = s"convAE_ds${data_size}_epoch${e}of${epoch}_batch$batch.txt"

        var ys = List[DenseVector[Int]]()
        for (z <- train_d) {
          val y = g.predict(z).map(o => (o * 256).toInt)
          ys = y :: ys
          g.reset()
        }

        // generate from Z space
        var zs = List[DenseVector[Int]]()
        for (z <- Z) {
          val zy = g.predict(z+train_d.head).map(o => (o * 256).toInt)
          zs = zy :: zs
          g.reset()
        }

        utils.write(s"${res_path}/${filename}", ys.reverse)
        utils.write(s"${res_path}/test_${filename}", test._2.map(i => convert(i * 256d, Int)))
        utils.write(s"${res_path}/zspace_${filename}", zs.reverse)

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
      //println(s"$e, $E")
    }

    println(args.toList)

    system.stop(err_actor)
    system.stop(img_actor)
    system.stop(test_img_actor)

    system.terminate()

    Thread.sleep(1000)
    System.exit(0)
  }

}
