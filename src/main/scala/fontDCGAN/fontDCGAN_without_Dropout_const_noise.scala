package fontDCGAN

import pll._
import breeze.linalg._

object atFontDCGAN_without_Dropout_const_noise {
  // parameter
  var TRAIN_DATA           = "data/fonts/font-all-d.txt"
  var DATA_SIZE            = 10000
  var TEST_DATA            = "/home/share/number/test-d.txt"
  var TEST_DATA_SIZE       = 10000
  var BATCH_SIZE           = 32
  var NUM_EPOCH            = 20
  var GENERATED_IMAGE_PATH = "src/main/scala/fontDCGAN/img"
  var LOAD_PARAM_G         = ""
  var LOAD_PARAM_D         = ""

  val D = Discriminator_without_Dropout() // Loss-func is 'cross entropy'
  val G = Generator()                     // Loss-func is 'cross entropy'

  def train(BATCH_SIZE: Int, NUM_EPOCH: Int)(TRAIN_DATA: String, DATA_SIZE: Int)(
      TEST_DATA: String,
      TEST_DATA_SIZE: Int)(GENERATED_IMAGE_PATH: String)(LOAD_PARAM_G: String,
                                                         LOAD_PARAM_D: String) = {

    println(
      s"load data from:\n\ttrain data:\t$TRAIN_DATA\n\ttest data:\t$TEST_DATA\ngen data:\t$GENERATED_IMAGE_PATH")
    println(s"\tnum of epoch:\t$NUM_EPOCH")

    println("--- loading ---")
    val train_d = utils.read(TRAIN_DATA, DATA_SIZE)
    // val test_d = utils.read(TEST_DATA, 100)(" ")
    if (LOAD_PARAM_G != "")
      G.model.load(LOAD_PARAM_G)
    if (LOAD_PARAM_D != "")
      D.model.load(LOAD_PARAM_D)
    println("--- load complete ---")

    println("--- train start ---")
    var (g_loss_placeholder, d_loss_placeholder) = (0d, 0d)

    val num_batches = (train_d.size / BATCH_SIZE).toInt
    println(s"Number of Batches: $num_batches")

    val noise = Array.fill(DATA_SIZE) { DenseVector.fill(100) { util.Random.nextDouble } }

    for (epoch <- 0 until NUM_EPOCH) {

      for (index <- 0 until num_batches) {

        val image_batch     = train_d.slice(index * BATCH_SIZE, (index + 1) * BATCH_SIZE)
        val generated_image = G.model.predict(noise.drop(index * BATCH_SIZE).take(BATCH_SIZE))
        G.model.reset()

        // TODO: Now this if-state is save data.
        //       Change behaver: save -> save and show
        if (index % 100 == 0 || index == num_batches - 1) {
          save_gen_imgs(generated_image, s"${GENERATED_IMAGE_PATH}/${epoch}_${index}.txt")
        }

        // Update Discriminator
        val X: Array[DenseVector[Double]] = image_batch ++ generated_image
        val y: Array[DenseVector[Double]] = Array.fill(BATCH_SIZE) { DenseVector(1d) } ++ Array
          .fill(BATCH_SIZE) { DenseVector(0d) }
        //val (d_loss, d_ys_list) = D.model.batch_train(X, y, /* 2 * */BATCH_SIZE, err.calc_cross_entropy_loss, grad.calc_cross_entropy_grad)

        val predicts = D.model.predict(X)
        val d_loss   = err.calc_cross_entropy_loss(predicts, y)
        val d_grads  = grad.calc_cross_entropy_grad(predicts, y)
        D.model.update(d_grads)

        // Update Generator

        val gen      = G.model.predict(noise.drop(index * BATCH_SIZE).take(BATCH_SIZE))
        val dis      = D.model.predict(gen)
        val g_loss   = err.calc_cross_entropy_loss(dis, Array.fill(BATCH_SIZE) { DenseVector(1d) })
        val gd_grads = grad.calc_cross_entropy_grad(dis, Array.fill(BATCH_SIZE) { DenseVector(1d) })
        val D_back   = D.model.backprop(gd_grads)
        D.model.reset()
        G.model.update(D_back)

        if (epoch == 0 && index == 0) {
          println("\nepoch,index,g_loss,d_loss")
        }
        println(s"$epoch, $index, $g_loss, $d_loss")
        g_loss_placeholder = g_loss
        d_loss_placeholder = d_loss
      }

      // save each network's parameters
      if (epoch % 10 == 0) {
        G.model.save_one_file(s"${GENERATED_IMAGE_PATH}/Gen_e${epoch}")
        D.model.save_one_file(s"${GENERATED_IMAGE_PATH}/Dis_e${epoch}")
      }

    }

    (g_loss_placeholder, d_loss_placeholder)

  }

  def save_gen_imgs(gen_imgs: Array[DenseVector[Double]], filename: String) = {
    val fos = new java.io.FileOutputStream(filename, false)
    val osw = new java.io.OutputStreamWriter(fos, "UTF-8")
    val pw  = new java.io.PrintWriter(osw)

    gen_imgs.foreach { i =>
      pw.write((i * 256d).toArray.mkString(","))
      pw.write("\n")
    }

    pw.close()
  }

  def data_load(): Array[DenseVector[Double]] = {
    utils.read(TRAIN_DATA, DATA_SIZE)
  }

  def args_process(args: Array[String]) {
    // process of arguments
    if (args.nonEmpty) {
      val opt_value = args.grouped(2)
      opt_value.foreach { ov =>
        ov(0) match {
          case "--train-path"     => TRAIN_DATA = ov(1)
          case "--data-size"      => DATA_SIZE = ov(1).toInt
          case "--test-path"      => TEST_DATA = ov(1)
          case "--test-data-size" => TEST_DATA_SIZE = ov(1).toInt
          case "--batch"          => BATCH_SIZE = ov(1).toInt
          case "--epoch"          => NUM_EPOCH = ov(1).toInt
          case "--save-path"      => GENERATED_IMAGE_PATH = ov(1)
          case "--load-param-g"   => LOAD_PARAM_G = ov(1)
          case "--load-param-d"   => LOAD_PARAM_D = ov(1)
          case _                  => println(s"option[${ov(0)}] is not exist.")
        }
      }
    }
  }

  def main(args: Array[String]) {

    args_process(args)

    train(BATCH_SIZE, NUM_EPOCH)(TRAIN_DATA, DATA_SIZE)(TEST_DATA, TEST_DATA_SIZE)(
      GENERATED_IMAGE_PATH)(LOAD_PARAM_G, LOAD_PARAM_D)

  }

}
