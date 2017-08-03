package fontGLO

import pll._
import breeze.linalg._
import scala.language.postfixOps

class font_batch_GLO(var LapDir: Int = 8) extends batchNet {// {{{

  // 100(1*10*10) -> 10*4*4 -> 5*8*8 -> 3*16*16 -> 1*32*32
/*  val pad = 1
  val fil_w = 2
  val stride = 1
  val up = "Adam"
  val lr = 0d
  layers =
    (new Pad(1, 1, "down")) ::
    (new Convolution(12, 3, 10, 1, 3, "He", 1d, up, lr)) ::
    (new LeakyReLU(0.01)) ::
    (new Pad(10,pad,"up")) ::
    (new Convolution(4*(pad+1)+pad,fil_w,5,10,stride,"He",1d,up,lr)) ::
    (new LeakyReLU(0.01)) ::
    (new Pad(5,pad,"up")) ::
    (new Convolution(8*(pad+1)+pad,fil_w,3,5,stride,"He",1d,up,lr)) ::
    (new LeakyReLU(0.01)) ::
    (new Pad(3,pad,"up")) ::
    (new Convolution(16*(pad+1)+pad,fil_w,1,3,stride,"He",1d,up,lr)) ::
    (new LeakyReLU(0.01)) ::
    layers
 */
  val Lap = Convolution(32,3,1,1,1,"",1d,"",1d)

  val l = LapDir match {
    case 4 => DenseVector[Double](
      0, 1, 0,
      1,-4, 1,
      0, 1, 0
    )
    case 8 | _ => DenseVector[Double](
      1, 1, 1,
      1,-8, 1,
      1, 1, 1
    )
  }
  Lap.F  = Array(Array(l))

  def calc_Lap_loss(y: DenseVector[Double], t: DenseVector[Double]) = {
    math.pow(2d,-2d) * sum((Lap.forward(y) - Lap.forward(t)).map(math.abs))
  }
  def calc_Lap_grad(y: DenseVector[Double], t: DenseVector[Double]) = {
    val o = Lap.forward(y) - Lap.forward(t)
    val W = Lap.filter2Weight(l, 32*32, 1)
    W.t * o.map(a => if(a>0) 1d else -1d)
  }

  def calc_Lap_loss(ys: Array[DenseVector[Double]], ts: Array[DenseVector[Double]]): Double = {
    var E = 0d
    for((y,t) <- ys zip ts){
      E += calc_Lap_loss(y,t)
    }
    E
  }

  def calc_Lap_grad(ys: Array[DenseVector[Double]], ts:Array[DenseVector[Double]]): Array[DenseVector[Double]] = {
    val grads = for((y,t) <- ys zip ts) yield { calc_Lap_grad(y,t) }
    grads.toArray
  }

}// }}}

object font_batch_GLO {

  def main(args: Array[String]){
    val rand = new util.Random(0)

    // parameters
    var zdim = 100
    var ds = 25
    var epoch = 100
    var batch = 5
    var pad = 1 //3
    var fil_w = 2 //4
    var stride = 1 //2
    var up = "Adam"
    var lr = 0d
    var Loss = "Laplacian"
    var LapDir = 8
    var doShuffle = true
    // doSave
    // saveTime

    var i = 0
    while(i < args.size){
      args(i) match {
        case "-z" | "--z-dimension" => { zdim = args(i+1).toInt; i+=2 }
        case "-d" | "--data-size" => { ds = args(i+1).toInt; i+=2 }
        case "-e" | "--epoch" => { epoch = args(i+1).toInt; i+=2 }
        case "-b" | "--batch-size" => { batch = args(i+1).toInt; i+=2 }
        case "-L" | "--loss-function" => { Loss = args(i+1); i+=2 }
        case        "--laplacian-filter-dir" => { LapDir = args(i+1).toInt; i+=2 }
        case "-p" | "--pad" => { pad = args(i+1).toInt; i+=2 }
        case "-f" | "--filter-width" => { fil_w = args(i+1).toInt; i+=2 }
        case "-s" | "--stride" => { stride = args(i+1).toInt; i+=2 }
        case "-u" | "--update-method" => { up = args(i+1); i+=2 }
        case "-l" | "--learning-rate" => { lr = args(i+1).toDouble; i+=2 }
        case "--do-shuffle" => { doShuffle = args(i+1).toBoolean; i+=2 }
        case _ => { println(s"unknown option:${args(i)}"); i+=1 }
      }
    }

    val start_time = (scala.sys.process.Process("date +%y%m%d-%H%M%S") !!).init
    val res_path = s"src/main/scala/fontGLO/results/${start_time}${args.mkString("-")}"
    val weights_path = s"src/main/scala/fontGLO/weights/${start_time}${args.mkString("-")}"
    val mkdir = scala.sys.process.Process(s"mkdir -p ${res_path} ${weights_path}").run
    mkdir.exitValue()

    var Z = Array.ofDim[DenseVector[Double]](ds)
    Z = Z.map(dv => DenseVector.fill(zdim){rand.nextGaussian / math.sqrt(ds)})
    val train_d = read("data/fonts/font-all-d.txt", ds)

    val g = new font_batch_GLO(LapDir)
    g.add(new Pad(1, 1, "down"))
      .add(new Convolution(12, 3, 10, 1, 3, "He", 1d, up, lr))
      //.add(new BatchNorm(10, 4*4))
      .add(new ReLU())
      .add(new Pad(10,pad,"up"))
      .add(new Convolution(4*(pad+1)+pad,fil_w,5,10,stride,"He",1d,up,lr))
      //.add(new BatchNorm(5, 8*8))
      .add(new ReLU())
      .add(new Pad(5,pad,"up"))
      .add(new Convolution(8*(pad+1)+pad,fil_w,3,5,stride,"He",1d,up,lr))
      //.add(new BatchNorm(3, 16*16))
      .add(new ReLU())
      .add(new Pad(3,pad,"up"))
      .add(new Convolution(16*(pad+1)+pad,fil_w,1,3,stride,"He",1d,up,lr))
      .add(new Tanh())

    // training
    for(e <- 0 until epoch){
      var E = 0d
      var unusedIdx =
        if(doShuffle) rand.shuffle(List.range(0,ds))
        else List.range(0,ds)

      while(unusedIdx.size != 0){
        val batchMask = unusedIdx take batch
        unusedIdx = unusedIdx drop batch

        val xs = (for(idx <- batchMask) yield { Z(idx) }).toArray
        val ts = (for(idx <- batchMask) yield { train_d(idx) }).toArray
        val ys = g.predict(xs)
        var d = Array[DenseVector[Double]]()
        Loss match {
          case "Laplacian" | "laplacian" | "Lap" | "lap" => {
            E += g.calc_Lap_loss(ys,ts)
            d = g.calc_Lap_grad(ys,ts)
          }
          case "L2" | _ => {
            E += g.calc_L2(ys,ts)
            d = g.calc_L2_grad(ys,ts)
          }
        }
        g.update(d)
      }

      // output
      if(e==0 || e%(epoch/10) == 0 || e == epoch - 1){
        val filename = s"batch_font_GLO_ds${ds}_epoch${e}of${epoch}_batch${batch}_pad${pad}_stride${stride}.txt"

        var ys = List[DenseVector[Int]]()
        for(z <- Z){
          val y = g.predict(z).map(o => (o*256).toInt)
          ys = y :: ys
          g.reset()
        }

        write(s"${res_path}/${filename}",ys.reverse)

        // save weights
        for(i <- 0 until g.layers.size){
          val LAYER = (g.layers(i).getClass()).toString.split(" ").last.drop(4)
          // MEMO:
          //   (g.layers(i).getClass()).toString ==> 'class pll.hogehoge'
          //   (g.layers(i).getClass()).toString.split(" ").last.drop(4) ==> 'hogehoge'

          g.layers(i).save(s"${weights_path}/${LAYER}_${i}_${filename}")
        }
      }

      println(s"$e, $E")
    }

    println(args.toList)
  }

  def read(fn: String, ds: Int = 100): Array[DenseVector[Double]] = { // {{{
    val f = io.Source.fromFile(fn).getLines.take(ds).map(_.split(",").map(_.toDouble/256d).toArray).toArray
    val g = f.map(a => DenseVector(a))
    g
  } // }}}

  def write(fn:String,dataList: List[DenseVector[Int]]){ // {{{
    val fos = new java.io.FileOutputStream(fn,false) //true: 追記, false: 上書き
    val osw = new java.io.OutputStreamWriter(fos,"UTF-8")
    val pw = new java.io.PrintWriter(osw)
    for(data <- dataList){
      for(i <- 0 until data.size){
        pw.write(data(i).toString)
        if(i != data.size -1){
          pw.write(",")
        }
      }
      pw.write("\n")
    }
    pw.close()
  } // }}}
}
