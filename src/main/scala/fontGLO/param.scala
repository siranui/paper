package fontGLO


import pll._

object param {
  // parameters
  var zdim = 100
  var data_size = 25
  var epoch = 100
  var batch = 5
  var pad = 1 //3
  var fil_w = 2 //4
  var stride = 1 //2
  var distr = "He"
  var SD = 1d
  var up = "Adam"
  var lr = 0.0002
  var Loss = "Laplacian"
  var LapDir = 8
  var doShuffle = true
  var doSave = true
  var doBatchNorm = true
  var saveTime = 10

  // for Network
  var Layers = List[List[String]]()
  var Pad = List[Int]()
  var Ch = List[Int]()
  var InW = List[Int]()
  var ud = List[String]()


  def setParamFromArgs(args: Array[String]): Unit = {
    var i = 0
    while (i < args.length) {
      args(i) match {
        case "-z" | "--z-dimension"   =>
          zdim = args(i + 1).toInt
          i += 2
        case "-d" | "--data-size"     =>
          data_size = args(i + 1).toInt
          i += 2
        case "-e" | "--epoch"         =>
          epoch = args(i + 1).toInt
          i += 2
        case "-b" | "--batch-size"    =>
          batch = args(i + 1).toInt
          i += 2
        case "--do-batch-norm"        =>
          doBatchNorm = args(i + 1).toBoolean
          i += 2
        case "-L" | "--loss-function" =>
          Loss = args(i + 1)
          i += 2
        case "--laplacian-filter-dir" =>
          LapDir = args(i + 1).toInt
          i += 2
        case "-p" | "--pad"           =>
          pad = args(i + 1).toInt
          i += 2
        case "-f" | "--filter-width"  =>
          fil_w = args(i + 1).toInt
          i += 2
        case "-s" | "--stride"        =>
          stride = args(i + 1).toInt
          i += 2
        case "--distr"                =>
          distr = args(i + 1)
          i += 2
        case "--standard-deviation"   =>
          SD = args(i + 1).toDouble
          i += 2
        case "-u" | "--update-method" =>
          up = args(i + 1)
          i += 2
        case "-l" | "--learning-rate" =>
          lr = args(i + 1).toDouble
          i += 2
        case "--do-shuffle"           =>
          doShuffle = args(i + 1).toBoolean
          i += 2
        case "--do-save"              =>
          doSave = args(i + 1).toBoolean
          i += 2
        case "--save-time"            =>
          saveTime = args(i + 1).toInt
          i += 2
        case _                        =>
          println(s"unknown option:${args(i)}")
          i += 1
      }
    }
    if (epoch < saveTime) saveTime = epoch
  }


  def readConf(file: String): Unit = {
    val f = io.Source.fromFile(file).getLines.toList.filter(_.take(2) != "//")
    f.foreach { line =>
      line.split(":")(0) match {
        case "Layers" => Layers = line.split(":")(1).split(";").map(_.split(",").toList).toList
        case "Pad"    => Pad = line.split(":")(1).split(",").map(_.toInt).toList
        case "Ch"     => Ch = line.split(":")(1).split(",").map(_.toInt).toList
        case "InW"    => InW = line.split(":")(1).split(",").map(_.toInt).toList
        case "ud"     => ud = line.split(":")(1).split(",").toList
      }
    }
  }

  def createNetwork(mode: Int = 0) = {
    // val net = mode match {
    //   case 0 => new Network()
    //   case 1 => new NetworkWithDropout()
    //   case 2 => new batchNet()
    //   case 3 => new batch_font_GLO(LapDir)
    // }

    val net = new batch_font_GLO(LapDir)

    for (i <- Layers.indices) {
      for (layer <- Layers(i)) {
        layer match {
          case "P" =>
            net.add(new Pad(Ch(i), Pad(i), ud(i)))
          case "C" =>
            if (ud(i) == "down") {
              net.add(new Convolution(InW(i) + (2 * Pad(i)), 3, Ch(i + 1), Ch(i), 3, distr, SD, up, lr))
            } else {
              net.add(
                new Convolution(InW(i) * (Pad(i) + 1) + Pad(i), fil_w, Ch(i + 1), Ch(i), stride, distr, SD, up, lr)
              )
            }
          case "B" =>
            net.add(new BatchNorm(up))
          case "R" =>
            net.add(new ReLU())
          case "T" =>
            net.add(new Tanh())
          case _   =>
        }
      }
    }

    net
  }

}
