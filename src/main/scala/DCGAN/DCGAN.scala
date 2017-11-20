package DCGAN

import pll._
import breeze.linalg._

case class DCGAN(G: Generator, D: Discriminator) {
  val model = new batchNet()
  model.layers = G.model.layers ++ D.model.layers
}

object atMNIST {

  val D = Discriminator() // Loss-func is 'cross entropy'
  val G = Generator() // Loss-func is 'cross entropy'
  val dcgan = DCGAN(G, D)

  def train(BATCH_SIZE: Int, NUM_EPOCH: Int)(TRAIN_DATA: String)(TEST_DATA: String)(GENERATED_IMAGE_PATH: String) = {

    println(s"load data from:\n\ttrain data:\t$TRAIN_DATA\n\ttest data:\t$TEST_DATA")
    val train_d = utils.read(TRAIN_DATA)
    val test_d = utils.read(TEST_DATA)

    println("train start")

    val num_batches = (train_d.size / BATCH_SIZE).toInt
    println(s"Number of Batches: $num_batches")

    for (epoch <- 0 until NUM_EPOCH) {

      for (index <- 0 until num_batches) {

        val noise = Array.fill(BATCH_SIZE){ DenseVector.fill(100){ util.Random.nextDouble } }
        val image_batch = train_d.slice(index * BATCH_SIZE, (index + 1) * BATCH_SIZE)
        val generated_image = G.model.predict(noise)

        // TODO: Now this if-state is save data.
        //       Change behaver: save -> save and show
        if (index % 500 == 0) {
          save_gen_imgs(generated_image, s"${GENERATED_IMAGE_PATH}/${epoch}_${index}.txt")
        }

        // Update Discriminator
        val X: Array[DenseVector[Double]] = image_batch ++ generated_image
        val y: Array[DenseVector[Double]] = Array.fill(BATCH_SIZE){ DenseVector.ones[Double](1) } ++ Array.fill(BATCH_SIZE){ DenseVector.zeros[Double](1) }
        val (d_loss, d_ys_list) = D.model.batch_train(X, y, 2 * BATCH_SIZE, err.calc_cross_entropy_loss, grad.calc_cross_entropy_grad)

        // Update Generator
        val noise2 = Array.fill(BATCH_SIZE){ DenseVector.fill(100){ util.Random.nextDouble } }
        val (g_loss, g_ys_list) = dcgan.model.batch_train(noise2, Array.fill(BATCH_SIZE){ DenseVector.ones[Double](1) }, BATCH_SIZE, err.calc_cross_entropy_loss, grad.calc_cross_entropy_grad)

        println((epoch, index, g_loss, d_loss))

      }

    }

  }

  def save_gen_imgs(gen_imgs: Array[DenseVector[Double]], filename: String) = {
    val fos = new java.io.FileOutputStream(filename, false)
    val osw = new java.io.OutputStreamWriter(fos, "UTF-8")
    val pw = new java.io.PrintWriter(osw)

    gen_imgs.foreach{ i =>
      pw.write((i * 256d).toArray.mkString(","))
      pw.write("\n")
    }

    pw.close()
  }

  def main(args: Array[String]) {

    var TRAIN_DATA = "/home/share/number/train-d.txt"
    var TEST_DATA = "/home/share/number/test-d.txt"
    var BATCH_SIZE = 32
    var NUM_EPOCH = 20
    var GENERATED_IMAGE_PATH = "src/main/scala/DCGAN/img"

    // process of arguments
    if (args.nonEmpty) {
      val opt_value = args.grouped(2)
      opt_value.foreach{ ov =>
        ov(0) match {
          case "--train-path" => TRAIN_DATA = ov(1)
          case "--test-path"  => TEST_DATA = ov(1)
          case "--batch"      => BATCH_SIZE = ov(1).toInt
          case "--epoch"      => NUM_EPOCH = ov(1).toInt
          case "--save-path"  => GENERATED_IMAGE_PATH = ov(1)
          case _              => println(s"option[${ov(0)}] is not exist.")
        }
      }
    }

    train(BATCH_SIZE, NUM_EPOCH)(TRAIN_DATA)(TEST_DATA)(GENERATED_IMAGE_PATH)

  }

}
