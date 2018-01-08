package fontGLO

// import breeze.linalg._
import pll.typeAlias._

object genTools {

  def Initialize(weightPath: String, zPath: String) = {
    val generator = fontDCGAN.Generator()(weightPath)
    val z = pll.utils.read(zPath, divV = 1)
    (generator, z)
  }

  def createInputsForMorphing(A: DV, B: DV)(implicit nrOfGenImg: Int = 100): Seq[DV] = {
    for {
      i <- 0 until nrOfGenImg
    } yield {
      A * (1d - i / nrOfGenImg.toDouble) + B * (i / nrOfGenImg.toDouble)
    }
  }

  def main(args: Array[String]) {
    var weightPath = ""
    var zPath = ""
    var morpA = 0
    var morpB = 1
    var savePath = s"${sys.env("HOME")}/morp.txt"
    var doSave = false

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
        case _ =>
          pll.log.error(s"${args(i)} is unknown option.")
          sys.exit(0)
      }
    }

    val (g, z) = Initialize(weightPath, zPath)

    val target = createInputsForMorphing(z(morpA), z(morpB))

    print("    ")
    val morpedTarget = target.map{v =>
      print(f"\b\b\b\b${target.indexOf(v)}%04d")
      g.model.predict(v)
    }

    if(doSave) {
      pll.utils.write(savePath, morpedTarget)
    }
  }

}
