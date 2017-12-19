package fontDCGAN

import pll._
import breeze.linalg._

case class DCGAN(G: Generator, D: Discriminator) {
  val model = new batchNet()
  model.layers = G.model.layers ++ D.model.layers
}

object atFont {

  val D     = Discriminator() // Loss-func is 'cross entropy'
  val G     = Generator() // Loss-func is 'cross entropy'
  val dcgan = DCGAN(G, D)

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

    val num_batches = (train_d.size / BATCH_SIZE).toInt
    println(s"Number of Batches: $num_batches")

    for (epoch <- 0 until NUM_EPOCH) {

      for (index <- 0 until num_batches) {

        val noise           = Array.fill(BATCH_SIZE) { DenseVector.fill(100) { util.Random.nextDouble } }
        val image_batch     = train_d.slice(index * BATCH_SIZE, (index + 1) * BATCH_SIZE)
        val generated_image = G.model.predict(noise)
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
        val (d_loss, d_ys_list) = D.model.batch_train(
          X,
          y, /* 2 * */ BATCH_SIZE,
          err.calc_cross_entropy_loss,
          grad.calc_cross_entropy_grad)
        // val (d_loss, d_ys_list) = D.model.batch_train(X, y, 2 * BATCH_SIZE, err.calc_L2, grad.calc_L2_grad)

        // Update Generator
        val noise2 = Array.fill(BATCH_SIZE) { DenseVector.fill(100) { util.Random.nextDouble } }
        val (g_loss, g_ys_list) = dcgan.model.batch_train(noise2, Array.fill(BATCH_SIZE) {
          DenseVector(1d)
        }, BATCH_SIZE, err.calc_cross_entropy_loss, grad.calc_cross_entropy_grad)
        // val (g_loss, g_ys_list) = dcgan.model.batch_train(noise2, Array.fill(BATCH_SIZE){ DenseVector.ones[Double](1) }, BATCH_SIZE, err.calc_L2, grad.calc_L2_grad)

        if (epoch == 0 && index == 0) {
          println("\nepoch,index,g_loss,d_loss")
        }
        println(s"$epoch, $index, $g_loss, $d_loss")

      }

      // save each network's parameters
      G.model.save_one_file(s"${GENERATED_IMAGE_PATH}/Gen_e${epoch}")
      D.model.save_one_file(s"${GENERATED_IMAGE_PATH}/Dis_e${epoch}")

    }

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

  def main(args: Array[String]) {

    var TRAIN_DATA           = "data/fonts/font-all-d.txt"
    var DATA_SIZE            = 10000
    var TEST_DATA            = "/home/share/number/test-d.txt"
    var TEST_DATA_SIZE       = 10000
    var BATCH_SIZE           = 32
    var NUM_EPOCH            = 20
    var GENERATED_IMAGE_PATH = "src/main/scala/fontDCGAN/img"
    var LOAD_PARAM_G         = ""
    var LOAD_PARAM_D         = ""

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

    train(BATCH_SIZE, NUM_EPOCH)(TRAIN_DATA, DATA_SIZE)(TEST_DATA, TEST_DATA_SIZE)(
      GENERATED_IMAGE_PATH)(LOAD_PARAM_G, LOAD_PARAM_D)

  }

}
