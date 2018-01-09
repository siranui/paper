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
  var dataSources     = List[String]()
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
        case "--data-sources"              =>
          i += 1
          while(i < args.size && args(i).head != '-'){
            dataSources = args(i) :: dataSources
            i += 1
          }
          dataSources = dataSources.reverse
        case "--do-shuffle"                =>
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
        case "-h" | "--help"               =>
          printHelp()
          sys.exit()
        case _                             =>
          println(s"unknown option:${args(i)}")
          i += 1
      }
    }
    if (epoch < saveTime) saveTime = epoch
  }

  val red = "\u001b[31m"
  val reset = "\u001b[0m"
  val depricated = s"${red}@depricated${reset}"

  def printHelp() {
    println(s"""
    [OPTION]
        -z | --z-dimension          set z's dimension.
        -d | --data-size            set using data size if you don't use '--data-sources' option.
        -e | --epoch                set training epoch.
        -b | --batch-size           set training batch size.
        -L | --loss-function        set loss function for training.
        --laplacian-filter-dir      $depricated: set Laplacian filter direction.
        --save-path                 set save directory path.
        -n | --network-conf-file    $depricated: set config file that writed network stracture
        --data-source               set data source.
                                    this option is low priority than '--data-sources' option.
        --data-sources              set data sources.
                                    this option is high priority than '--data-source' option.
        --do-shuffle                set data shuffle flag. if this flag is 'true', do shuffle while training.
        --do-save                   set save flag.
        --save-time                 set num of excute-save-process.
        --display                   $depricated: set flag of display.
        --load-param-g              set path for load Generator parameters.
        --load-z                    set path for load Z.
        --distr                     set distribution for training model.
        --standard-deviation        set standard-deviation for training model.
        --update-method             set update-method for training model.
        --learning-rate             set learning-rate for training model.
        -h | --help                 print this help.

    [DEFAULT PARAMETERS]
        zdim            = $zdim
        data_size       = $data_size
        epoch           = $epoch
        batch           = $batch
        Loss            = $Loss
        LapDir          = $LapDir
        savePath        = $savePath
        networkConfFile = $networkConfFile
        dataSource      = $dataSource
        dataSources     = $dataSources
        doShuffle       = $doShuffle
        doSave          = $doSave
        display         = $display
        saveTime        = $saveTime
      
        distr           = $distr
        SD              = $SD
        update_method   = $update_method
        lr              = $lr
      
        LOAD_PARAM_G    = $LOAD_PARAM_G
        LOAD_Z          = $LOAD_Z
      
    """
    )
  }


}
