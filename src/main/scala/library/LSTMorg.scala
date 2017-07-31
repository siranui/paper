package pll
import breeze.linalg._
import breeze.numerics.{sigmoid, tanh}

class LSTMorg(val xn:Int,val hn:Int,val dist:String,var n:Double,val u:String,val a:Double) extends Layer{

  var Ft = List[DenseVector[Double]]()
  var It = List[DenseVector[Double]]()
  var Cc = List[DenseVector[Double]]()
  var Ot = List[DenseVector[Double]]()
  var Co = List[DenseVector[Double]]()

  var z = List[DenseVector[Double]]()

  var Cr = List[DenseVector[Double]]()
  var Hr = List[DenseVector[Double]]()

//  val rand = new util.Random(0)
  var Wf = DenseMatrix.zeros[Double](hn,xn+hn)
  var Wi = DenseMatrix.zeros[Double](hn,xn+hn)
  var Wc = DenseMatrix.zeros[Double](hn,xn+hn)
  var Wo = DenseMatrix.zeros[Double](hn,xn+hn)

  if(dist == "Gaussian"){
    Wf = Gaussian(hn,hn+xn,hn+xn)
    Wi = Gaussian(hn,hn+xn,hn+xn)
    Wc = Gaussian(hn,hn+xn,hn+xn)
    Wo = Gaussian(hn,hn+xn,hn+xn)
  }else if(dist == "Uniform"){
    Wf = Uniform(hn,hn+xn,hn+xn)
    Wi = Uniform(hn,hn+xn,hn+xn)
    Wc = Uniform(hn,hn+xn,hn+xn)
    Wo = Uniform(hn,hn+xn,hn+xn)
  }else if(dist == "Xavier"){
    Wf = Xavier(hn,hn+xn,hn+xn)
    Wi = Xavier(hn,hn+xn,hn+xn)
    Wc = Xavier(hn,hn+xn,hn+xn)
    Wo = Xavier(hn,hn+xn,hn+xn)
  }else if(dist == "He"){
    Wf = He(hn,hn+xn,hn+xn)
    Wi = He(hn,hn+xn,hn+xn)
    Wc = He(hn,hn+xn,hn+xn)
    Wo = He(hn,hn+xn,hn+xn)
  }
      /*
       var Wf = DenseMatrix.fill(hn, xn + hn) {
       rand.nextGaussian / math.sqrt(xn + hn)
       }
       var Wi = DenseMatrix.fill(hn, xn + hn) {
       rand.nextGaussian / math.sqrt(xn + hn)
       }
       var Wc = DenseMatrix.fill(hn, xn + hn) {
       rand.nextGaussian / math.sqrt(xn + hn)
       }
       var Wo = DenseMatrix.fill(hn, xn + hn) {
       rand.nextGaussian / math.sqrt(xn + hn)
       }
       */
  var bf = DenseVector.zeros[Double](hn)
  var bi = DenseVector.zeros[Double](hn)
  var bc = DenseVector.zeros[Double](hn)
  var bo = DenseVector.zeros[Double](hn)

  var dWf = DenseMatrix.zeros[Double](hn, xn + hn)
  var dWi = DenseMatrix.zeros[Double](hn, xn + hn)
  var dWc = DenseMatrix.zeros[Double](hn, xn + hn)
  var dWo = DenseMatrix.zeros[Double](hn, xn + hn)

  var dbf = DenseVector.zeros[Double](hn)
  var dbi = DenseVector.zeros[Double](hn)
  var dbc = DenseVector.zeros[Double](hn)
  var dbo = DenseVector.zeros[Double](hn)

  var dC = DenseVector.zeros[Double](hn)
  var dN = DenseVector.zeros[Double](hn)


  var opt = Opt.create(u,a)
  opt.register(Array(bf,bi,bc,bo))
  opt.register(Array(Wf,Wi,Wc,Wo))

  def forward(x: DenseVector[Double]): DenseVector[Double] = {
    z = Hr.size match {
      case 0 => DenseVector.vertcat(DenseVector.zeros[Double](hn), x) :: z
      case _ => DenseVector.vertcat(Hr.head, x) :: z
    }

    Ft = sigmoid(Wf * z.head + bf) :: Ft
    It = sigmoid(Wi * z.head + bi) :: It
    Cc = tanh(Wc * z.head + bc) :: Cc
    Ot = sigmoid(Wo * z.head + bo) :: Ot

    Cr = Cr.size match {
      case 0 => (Ft.head + (It.head *:* Cc.head)) :: Cr
      case _ => ((Cr.head *:* Ft.head) + (It.head *:* Cc.head)) :: Cr
    }

    Co = tanh(Cr.head) :: Co

    Hr = (Ot.head *:* Co.head) :: Hr
    Hr.head
  }

  def forward(x: DenseVector[Double], H: DenseVector[Double], C: DenseVector[Double]) : DenseVector[Double] ={
    Hr = H :: Hr
    Cr = C :: Cr
    forward(x)
  }
  
  def HRCR()={
    (Hr.head, Cr.head)
  }

  def backward(dh: DenseVector[Double]): DenseVector[Double] = {
    val dj = dh + dN
    val dm = (dj *:* Co.head) *:* (Ot.head *:* (1d - Ot.head))
    val dk = (dh *:* Ot.head) *:* (1d - (Co.head *:* Co.head)) + dC
    val dp = (dk *:* It.head) *:* (1d - (Cc.head *:* Cc.head))
    val dq = (dk *:* Cc.head) *:* It.head *:* (1d - It.head)
    val ds = (dk *:* Cr.head) *:* Ft.head *:* (1d - Ft.head)

    val dz = (Wf.t * ds) + (Wi.t * dq) + (Wc.t * dp) + (Wo.t * dm)


    // 前の時刻へのデルタの更新
    dN = dz(0 until hn)
    dC = dk *:* Ft.head

    // 重みの更新量の更新
    dWf += ds * z.head.t
    dWi += dq * z.head.t
    dWc += dp * z.head.t
    dWo += dm * z.head.t

    // バイアスの更新量の更新
    dbf += ds
    dbi += dq
    dbc += dp
    dbo += dm

    Ft = Ft.tail
    It = It.tail
    Cc = Cc.tail
    Ot = Ot.tail
    Co = Co.tail

    z = z.tail

    Cr = Cr.tail
    Hr = Hr.tail

    dz(hn until dz.size)
  }

  def backward(dh: DenseVector[Double], N: DenseVector[Double], C: DenseVector[Double]) : DenseVector[Double] = {
    dN = N
    dC = C
    backward(dh)
  }

  def DNDC() = {
    (dN, dC)
  }


  def update(): Unit = {
    val dbs = opt.update(Array(bf,bi,bc,bo),Array(dbf,dbi,dbc,dbo))
    bf -= dbs(0)
    bi -= dbs(1)
    bc -= dbc(2)
    bo -= dbo(3)

    val dws = opt.update(Array(Wf,Wi,Wc,Wo),Array(dWf,dWi,dWc,dWo))
    Wf -= dws(0)
    Wi -= dws(1)
    Wc -= dws(2)
    Wo -= dws(3)
/*
    Wf -= learningRate *:* dWf
    Wi -= learningRate *:* dWi
    Wc -= learningRate *:* dWc
    Wo -= learningRate *:* dWo

    bf -= learningRate * dbf
    bi -= learningRate * dbi
    bc -= learningRate * dbc
    bo -= learningRate * dbo
 */
    reset()
  }

  def reset(): Unit = {
    Ft = List[DenseVector[Double]]()
    It = List[DenseVector[Double]]()
    Cc = List[DenseVector[Double]]()
    Ot = List[DenseVector[Double]]()
    Co = List[DenseVector[Double]]()

    z = List[DenseVector[Double]]()

    Cr = List[DenseVector[Double]]()
    Hr = List[DenseVector[Double]]()

    dWf = DenseMatrix.zeros[Double](hn, xn + hn)
    dWi = DenseMatrix.zeros[Double](hn, xn + hn)
    dWc = DenseMatrix.zeros[Double](hn, xn + hn)
    dWo = DenseMatrix.zeros[Double](hn, xn + hn)

    dbf = DenseVector.zeros[Double](hn)
    dbi = DenseVector.zeros[Double](hn)
    dbc = DenseVector.zeros[Double](hn)
    dbo = DenseVector.zeros[Double](hn)

    dC = DenseVector.zeros[Double](hn)
    dN = DenseVector.zeros[Double](hn)
  }
  def save(filename:String){


  }
  def load(data: List[String]) = {
    data
  }
}
