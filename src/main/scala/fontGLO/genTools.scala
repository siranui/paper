package fontGLO

// import breeze.linalg._
import pll._
import pll.typeAlias._

object genTools {
  var weightPath: String  = null
  var zPath: String       = null
  var dataSources         = List[String]()
  var morpA               = 0
  var morpB               = 1
  var savePath            = s"${sys.env("HOME")}/morp.txt"
  var behavior            = ""
  var newFontPath: String = null
  var split_ratio         = (0.5: T)
  var epoch               = 100
  var doSave              = false

  def Initialize(weightPath: String, zPath: String) = {
    val generator = fontDCGAN.Generator()(weightPath)
    val z         = pll.utils.read(zPath, divV = 1)
    (generator, z)
  }

  def createInputsForMorphing(A: DV, B: DV)(implicit nrOfGenImg: Int = 100): Seq[DV] = {
    for {
      i <- 0 until nrOfGenImg
    } yield {
      A * (1d - i / (nrOfGenImg: T)) + B * (i / (nrOfGenImg: T))
    }
  }

  def doMorphing(g: Network, z: Seq[DV])(implicit nrOfGenImg: Int = 100) = {

    val target = createInputsForMorphing(z(morpA), z(morpB))(nrOfGenImg)

    print("\n    ")
    val morpedTarget = target.map { v =>
      val y = g.predict(v)
      print(f"\b\b\b\b${target.indexOf(v)}%04d")
      g.reset()
      y
    }

    if (doSave) {
      pll.utils.write(savePath, morpedTarget)
    }

  }

  def trainFeatureModel(font_data: Seq[DV], font_feature: Seq[DV])(
      test_font_data: Seq[DV],
      test_font_feature: Seq[DV])(implicit epoch: Int = 100): Network = {
    val net = new batchNet
    // net.add(new Affine(32 * 32, 300, "He", 1d, "SGD", 0.01))
    // net.add(new Tanh())
    // net.add(new Dropout(0.5) )
    // net.add(new Affine(300, 300, "He", 1d, "SGD", 0.01))
    // net.add(new Tanh())
    // net.add(new Dropout(0.5) )
    // net.add(new Affine(300, 100, "He", 1d, "SGD", 0.01))
    // net.add(new Tanh())

    net.add(new i2cConv(32, 5, 8, 1, 3, "He", 1d, "Adam", 0d))
    net.add(new LeakyReLU())
    net.add(new Dropout(0.3))
    net.add(new Pooling(2, 2)(8, 10, 10))
    net.add(new i2cConv(5, 3, 32, 8, 1, "He", 1d, "Adam", 0d))
    net.add(new LeakyReLU())
    net.add(new Dropout(0.3))
    net.add(new Affine(3 * 3 * 32, 100, "He", 1d, "Adam", 0d))
    net.add(new LeakyReLU())
    net.add(new Dropout(0.3))
    net.add(new Affine(100, 100, "He", 1d, "Adam", 0d))
    net.add(new Tanh())

    log.info(s"train_data: ${font_data.size}")
    log.info(s"test_data: ${test_font_data.size}")
    log.info(s"data_dim: ${font_data(0).size}")
    log.info(s"feature_dim: ${font_feature(0).size}")
    log.info(s"epoch: ${epoch}")

    log.info(" *** training start *** ")

    for (e <- 0 until epoch) {
      var error     = 0d
      var testerror = 0d
      for ((x, t) <- (font_data zip font_feature)) {
        val y = net.predict(x)
        error += err.calc_L2(y, t)
        val d = grad.calc_L2_grad(y, t)
        net.update(d)
      }
      for ((x, t) <- (test_font_data zip test_font_feature)) {
        val y = net.forward_at_test(x)
        testerror += err.calc_L2(y, t)
        net.reset()
      }
      log.info(
        s"epoch: ${e}, error: ${error / font_data.size}, testerror: ${testerror / test_font_data.size}")
    }

    log.info(" *** training finis *** ")

    net
  }

  def createNewFeatureVector(inputs: Seq[DV])(model: Network): Seq[DV] = {

    inputs.map { v =>
      model.predict(v)
    }
  }

  def createNewFeatureVectorAndModel(inputs: Seq[DV],
                                     split_ratio: T = (0.5: T),
                                     epoch: Int = 100) = {
    val font_data: Array[DV] = dataSources match {
      case Nil =>
        log.error("data is nothing. please setting '--data-sources' option.")
        sys.exit(1)
      case _ =>
        dataSources.map(d => utils.read(d)).reduce(_ ++ _)
    }

    val font_feature: Array[DV] = Option(zPath) match {
      case Some(z) => pll.utils.read(z, divV = 1)
      case None =>
        log.error(s"'--z-path' option does not set.")
        sys.exit(3)
    }

    val dataset: Array[(DV, DV)] = font_data zip font_feature

    val tmp = utils.rand.shuffle(dataset.toList)

    val train_set = tmp.take((tmp.size * split_ratio).toInt)
    val test_set  = tmp.drop((tmp.size * split_ratio).toInt)

    val model = trainFeatureModel(train_set.map(_._1), train_set.map(_._2))(
      test_set.map(_._1),
      test_set.map(_._2))(epoch)

    val new_feature = createNewFeatureVector(inputs)(model)

    if (doSave) {
      pll.utils.write(s"${sys.env("HOME")}/new_feature.txt", new_feature)
      model.save_one_file(s"${sys.env("HOME")}/feature_model.weight")
    }
  }

  def main(args: Array[String]) {
    var i = 0
    while (i < args.size) {
      args(i) match {
        case "--weight-path" =>
          weightPath = args(i + 1)
          i += 2
        case "--z-path" =>
          zPath = args(i + 1)
          i += 2
        case "--data-sources" =>
          i += 1
          while (i < args.size && args(i).head != '-') {
            dataSources = args(i) :: dataSources
            i += 1
          }
          dataSources = dataSources.reverse
        case "--new-font-path" =>
          newFontPath = args(i + 1)
          i += 2
        case "--split-ratio" =>
          split_ratio = (args(i + 1).toDouble: T)
          i += 2
        case "--epoch" =>
          epoch = args(i + 1).toInt
          i += 2
        case "--morp-A" =>
          morpA = args(i + 1).toInt
          i += 2
        case "--morp-B" =>
          morpB = args(i + 1).toInt
          i += 2
        case "--do-save" =>
          doSave = args(i + 1).toBoolean
          i += 2
        case "--behavior" | "--do" =>
          behavior = args(i + 1)
          i += 2
        case _ =>
          pll.log.error(s"${args(i)} is unknown option.")
          sys.exit(0)
      }
    }

    // val (g, z) = Initialize(weightPath, zPath)

    behavior match {
      case "morphing" | "Morphing" =>
        assert(weightPath != null, "Set '--weight-path' option.")
        assert(zPath != null, "Set '--z-path' option.")

        val (g, z) = Initialize(weightPath, zPath)
        doMorphing(g.model, z)
      case "feature" =>
        val inputs = Option(newFontPath) match {
          case Some(f) => pll.utils.read(f)
          case None =>
            pll.log.error("Set '--new-font-path' option.")
            sys.exit(2)
        }

        createNewFeatureVectorAndModel(inputs, split_ratio, epoch)
      case _ =>
        log.error(
          s"'${behavior}' does not exist in the '--behavior' option. we assume 'morphing' or 'feature'.")
        sys.exit(1)
    }

  }

}
