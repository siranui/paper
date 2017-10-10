package pll

object set {

  var Layers = List[List[String]]()
  var Pad = List[Int]()
  var Ch = List[Int]()
  var InW = List[Int]()
  var UpDown = List[String]()
  var InitWeights = List[String]()
  var SDs = List[Double]()
  var UpdateMethod = List[String]()
  var lrs = List[Double]()
  var Filter_ws = List[Int]()
  var Strides = List[Int]()

  def main(args: Array[String]) {

    readConf(args(0))

    val net = connectNetwork(new Network())
    net.layers.foreach(println)

  }

  def setCheck(args: Array[String]) = {

    readConf(args(0))
    readConfCheck()
    connectNetwork(new batchNet())

  }

  def readConf(file: String): Unit = {
    val f = io.Source.fromFile(file).getLines.toList.filter(_.length != 0).filter(_.take(2) != "//")
    f.foreach { line =>
      line.split(":")(0).trim match {
        case "Layers" => Layers = line.split(":")(1).split(";").map(_.split(",").map(_.trim).toList).toList
        case "Pad"    => Pad = line.split(":")(1).split(",").map(_.trim.toInt).toList
        case "Ch"     => Ch = line.split(":")(1).split(",").map(_.trim.toInt).toList
        case "InW"    => InW = line.split(":")(1).split(",").map(_.trim.toInt).toList
        case "UpDown"     => UpDown = line.split(":")(1).split(",").map(_.trim).toList
        case "InitWeights"     => InitWeights = line.split(":")(1).split(",").map(_.trim).toList
        case "SDs"     => SDs = line.split(":")(1).split(",").map(_.trim.toDouble).toList
        case "UpdateMethod"     => UpdateMethod = line.split(":")(1).split(",").map(_.trim).toList
        case "lrs"     => lrs = line.split(":")(1).split(",").map(_.trim.toDouble).toList
        case "Filter_ws"     => Filter_ws = line.split(":")(1).split(",").map(_.trim.toInt).toList
        case "Strides"     => Strides = line.split(":")(1).split(",").map(_.trim.toInt).toList
      }
    }
  }

  def readConfCheck() = {
    println(Layers)
    println(Pad)
    println(Ch)
    println(InW)
    println(UpDown)
    println(InitWeights)
    println(SDs)
    println(UpdateMethod)
    println(lrs)
    println(Filter_ws)
    println(Strides)
  }

  def connectNetwork[N <: Network](net: N): N = {

    for(i <- Layers.indices){
      for(layer <- Layers(i)){
        layer match {
          case "P" =>
            net.add(new Pad(Ch(i),Pad(i),UpDown(i)))
          case "C" =>
            if(UpDown(i) == "down"){
              net.add(new Convolution(InW(i) + (2 * Pad(i)), Filter_ws(i), Ch(i+1), Ch(i), Strides(i), InitWeights(i), SDs(i), UpdateMethod(i), lrs(i)))
            } else {
              net.add(new Convolution(InW(i) * (Pad(i) + 1) + Pad(i), Filter_ws(i), Ch(i+1), Ch(i), Strides(i), InitWeights(i), SDs(i), UpdateMethod(i), lrs(i)))
            }
          case "BN" =>
            net.add(new BatchNorm(UpdateMethod(i)))
          case "R" =>
            net.add(new ReLU())
          case "LR" =>
            net.add(new LeakyReLU(0.02))
          case "T" =>
            net.add(new Tanh())
          case "S" =>
            net.add(new Sigmoid())
          case _   =>
        }
      }
      println("debug: " + net.layers(i))
    }

    net

  }

}
