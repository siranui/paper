package fontGLO

// import breeze.linalg._
import pll._
import pll.typeAlias._

object genTools {
  var weightPath = ""
  var zPath = ""
  var morpA = 0
  var morpB = 1
  var savePath = s"${sys.env("HOME")}/morp.txt"
  var behavior = ""
  var doSave = false


  def Initialize(weightPath: String, zPath: String) = {
    val generator = fontDCGAN.Generator()(weightPath)
    val z = pll.utils.read(zPath, divV = 1)
    (generator, z)
  }

  def createInputsForMorphing(A: DV, B: DV)(implicit nrOfGenImg: Int = 100): Seq[DV] = {
    for {
      i <- 0 until nrOfGenImg
    } yield {
      A * (1d - i / (nrOfGenImg:T)) + B * (i / (nrOfGenImg:T))
    }
  }

  def doMorphing(g: Network, z: Seq[DV])(implicit nrOfGenImg: Int = 100) = {
    
    val target = createInputsForMorphing(z(morpA), z(morpB))(nrOfGenImg)

    print("\n    ")
    val morpedTarget = target.map{v =>
      val y = g.predict(v)
      print(f"\b\b\b\b${target.indexOf(v)}%04d")
      g.reset()
      y
    }

    if(doSave) {
      pll.utils.write(savePath, morpedTarget)
    }

  }

  def trainFeatureModel(font_data: Seq[DV], font_feature: Seq[DV])(test_font_data: Seq[DV], test_font_feature: Seq[DV])(implicit epoch: Int = 1000): Network = {
    val net = new batchNet
    net.add( new Affine(32*32, 500, "He", 1d, "Adam", 0d) )
    net.add( new LeakyReLU() )
    net.add( new Affine(500, 500, "He", 1d, "Adam", 0d) )
    net.add( new LeakyReLU() )
    net.add( new Affine(500, 100, "He", 1d, "Adam", 0d) )
    net.add( new Tanh() )

    log.info(s"train_data: ${font_data.size}")
    log.info(s"test_data: ${test_font_data.size}")
    log.info(s"data_dim: ${font_data(0).size}")
    log.info(s"feature_dim: ${font_feature(0).size}")
    log.info(s"epoch: ${epoch}")

    for(e <- 0 until epoch){
      var error = 0d
      var testerror = 0d
      for( (x, t) <- (font_data zip font_feature) ) {
        val y = net.predict(x)
        error += err.calc_L2(y, t)
        val d = grad.calc_L2_grad(y, t)
        net.update(d)
      }
      for( (x, t) <- (test_font_data zip test_font_feature) ) {
        val y = net.predict(x)
        testerror += err.calc_L2(y, t)
        net.reset()
      }
      log.info(s"epoch: ${epoch}, error: ${error / font_data.size}, testerror: ${testerror / test_font_data.size}")
    }
    net
  }

  def createNewFeatureVector(inputs: Seq[DV])(model: Network): Seq[DV] = {

    inputs.map{v => model.predict(v) }
  }

  def main(args: Array[String]) {
    var i = 0
    while(i < args.size){
      args(i) match {
        case "--weight-path" =>
          weightPath = args(i+1)
          i += 2
        case "--z-path" =>
          zPath = args(i+1)
          i += 2
        case "--morp-A" =>
          morpA = args(i+1).toInt
          i += 2
        case "--morp-B" =>
          morpB = args(i+1).toInt
          i += 2
        case "--do-save" =>
          doSave = args(i+1).toBoolean
          i += 2
        case "--behavior" | "--do" =>
          behavior = args(i+1)
          i += 2
        case _ =>
          pll.log.error(s"${args(i)} is unknown option.")
          sys.exit(0)
      }
    }

    val (g, z) = Initialize(weightPath, zPath)

    
    behavior match {
      case "morphing" | "Morphing" =>
        doMorphing(g.model, z)
      case _ =>
        log.error(s"'${behavior}' is not exist. we assume 'morphing'")
        sys.exit(1)
    }

  }

}
