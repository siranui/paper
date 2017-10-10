package fontGLO


import pll._

object param {
  // parameters
  var zdim = 100
  var data_size = 25
  var epoch = 100
  var batch = 5
  var Loss = "Laplacian"
  var LapDir = 8
  var savePath = "src/main/scala/fontGLO"
  var networkConfFile = "net.conf"
  var dataSource = "/home/pll03/sbt/paper/data/fonts/font-all-d.txt"
  var doShuffle = true
  var doSave = true
  var saveTime = 10


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
        case "-L" | "--loss-function" =>
          Loss = args(i + 1)
          i += 2
        case "--laplacian-filter-dir" =>
          LapDir = args(i + 1).toInt
          i += 2
        case "--save-path"            =>
          savePath = args(i + 1)
          i += 2
        case "-n" | "--network-conf-file"  =>
          networkConfFile = args(i + 1)
          i += 2
        case "--data-source"  =>
          dataSource = args(i + 1)
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


}
