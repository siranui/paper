package fontGLO

object param {
  // parameters
  var zdim            = 100
  var data_size       = 25
  var epoch           = 100
  var batch           = 5
  var Loss            = "Laplacian"
  var LapDir          = 8
  var savePath        = "src/main/scala/fontGLO"
  var networkConfFile = "net.conf"
  var dataSource      = "data/fonts/font-all-d.txt"
  var doShuffle       = true
  var doSave          = true
  var display         = false
  var saveTime        = 10

  var distr         = "He"
  var SD            = 0.01
  var update_method = "Adam,0.5,0.999,1e-8"
  var lr            = 2e-4

  var LOAD_PARAM_G = ""
  var LOAD_Z       = ""

  def setParamFromArgs(args: Array[String]): Unit = {
    var i = 0
    while (i < args.length) {
      args(i) match {
        case "-z" | "--z-dimension" =>
          zdim = args(i + 1).toInt
          i += 2
        case "-d" | "--data-size" =>
          data_size = args(i + 1).toInt
          i += 2
        case "-e" | "--epoch" =>
          epoch = args(i + 1).toInt
          i += 2
        case "-b" | "--batch-size" =>
          batch = args(i + 1).toInt
          i += 2
        case "-L" | "--loss-function" =>
          Loss = args(i + 1)
          i += 2
        case "--laplacian-filter-dir" =>
          LapDir = args(i + 1).toInt
          i += 2
        case "--save-path" =>
          savePath = args(i + 1)
          i += 2
        case "-n" | "--network-conf-file" =>
          networkConfFile = args(i + 1)
          i += 2
        case "--data-source" =>
          dataSource = args(i + 1)
          i += 2
        case "--do-shuffle" =>
          doShuffle = args(i + 1).toBoolean
          i += 2
        case "--do-save" =>
          doSave = args(i + 1).toBoolean
          i += 2
        case "--save-time" =>
          saveTime = args(i + 1).toInt
          i += 2
        case "--display" =>
          display = args(i + 1).toBoolean
          i += 2
        case "--load-param-g" =>
          LOAD_PARAM_G = args(i + 1)
          i += 2
        case "--load-z" =>
          LOAD_Z = args(i + 1)
          i += 2
        case "--distr" =>
          distr = args(i + 1)
          i += 2
        case "--standard-deviation" =>
          SD = args(i + 1).toDouble
          i += 2
        case "--update-method" =>
          update_method = args(i + 1)
          i += 2
        case "--learning-rate" =>
          lr = args(i + 1).toDouble
          i += 2
        case _ =>
          println(s"unknown option:${args(i)}")
          i += 1
      }
    }
    if (epoch < saveTime) saveTime = epoch
  }

}
