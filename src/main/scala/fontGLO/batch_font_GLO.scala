package font_GLO

import pll._
import breeze.linalg._

class font_batch_GLO(var LapDir: Int = 8) extends batchNet {// {{{

  //
  // case class Pad(channel: Int, width: Int, ud: String = "down") extends Layer{
  //
  // case class Convolution(
  //   input_width: Int,
  //   filter_width: Int,
  //   filter_set: Int = 1,
  //   channel: Int = 1,
  //   stride: Int = 1,
  //   distr: String = "Gaussian",
  //   SD: Double = 1d,
  //   update_method: String = "SGD",
  //   lr: Double = 0.01,
  // ) extends Layer {
  // out_width = (math.floor((input_width - filter_width) / stride) + 1).toInt
  //class BatchNorm(ch:Int = 1, dim: Int, update_method: String = "SGD", lr: Double = 0.01) extends Layer {

  //// 100(1*10*10) -> 25*2*2 -> 10*4*4 -> 5*8*8 -> 3*16*16 -> 1*32*32
  // val pad = 1//3
  // val fil_w = 2//4
  // val stride = 1//2
  // val up = "Adam"
  // val lr = 0d
  // layers =
  //   (new Pad(25,pad,"up")) ::
  //   (new Convolution(2*(pad+1)+pad,fil_w,10,25,stride,"He",1d,up,lr)) ::
  //   // (new BatchNorm(10,4*4,up,lr)) ::
  //   (new LeakyReLU(0.01)) ::
  //   (new Pad(10,pad,"up")) ::
  //   (new Convolution(4*(pad+1)+pad,fil_w,5,10,stride,"He",1d,up,lr)) ::
  //   // (new BatchNorm(5,8*8,up,lr)) ::
  //   (new LeakyReLU(0.01)) ::
  //   (new Pad(5,pad,"up")) ::
  //   (new Convolution(8*(pad+1)+pad,fil_w,3,5,stride,"He",1d,up,lr)) ::
  //   // (new BatchNorm(3,16*16,up,lr)) ::
  //   (new LeakyReLU(0.01)) ::
  //   (new Pad(3,pad,"up")) ::
  //   (new Convolution(16*(pad+1)+pad,fil_w,1,3,stride,"He",1d,up,lr)) ::
  //   // (new BatchNorm(1,32*32,up,lr)) ::
  //   (new LeakyReLU(0.01)) ::
  //   layers

  // 100(1*10*10) -> 10*4*4 -> 5*8*8 -> 3*16*16 -> 1*32*32
  val pad = 1
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
    val zdim = 100
    val ds = 25
    val epoch = 5000
    val batch = 5
    // Loss(L2 or Lap)
    // LapDir(4 or 8)
    // doShuffle(true or False)
    // doSave
    // saveTime

    val start_time = (scala.sys.process.Process("date +%y%m%d-%H%M%S") !!).init
    val res_path = s"src/main/scala/fontGLO/results/${start_time}"
    val weights_path = s"src/main/scala/fontGLO/weights/${start_time}"
    val mkdir = scala.sys.process.Process(s"mkdir -p ${res_path} ${weights_path}").run
    mkdir.exitValue()

    var Z = Array.ofDim[DenseVector[Double]](ds)
    Z = Z.map(dv => DenseVector.fill(zdim){rand.nextGaussian})
    val train_d = read("/home/yuya/fonts/tmp/AGENCYB/AGENCYB-d.txt", ds)


    val g = new font_batch_GLO(LapDir=8)

    // training
    for(e <- 0 until epoch){
      var E = 0d
      var unusedIdx = rand.shuffle(List.range(0,ds))

      while(unusedIdx.size != 0){
        val batchMask = unusedIdx take batch
        unusedIdx = unusedIdx drop batch

        val xs = (for(idx <- batchMask) yield { Z(idx) }).toArray
        val ts = (for(idx <- batchMask) yield { train_d(idx) }).toArray
        val ys = g.predict(xs)
        E += g.calc_Lap_loss(ys,ts)
        val d = g.calc_Lap_grad(ys,ts)
        g.update(d)
      }

      // output
      if(e==0 || e%(epoch/10) == 0 || e == epoch - 1){
        val filename = s"batch_font_GLO_ds${ds}_epoch${e}of${epoch}_batch${batch}_pad${g.pad}_stride${g.stride}.txt"

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
