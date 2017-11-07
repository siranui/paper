package pll
import breeze.linalg._

class BNL(var xn:Int,var bn:Int)extends Layer{
  var beta = DenseVector.zeros[Double](xn)
  var dbeta = DenseVector.zeros[Double](xn)
  var gamma = DenseVector.ones[Double](xn)
  gamma = Gaussian(xn,0.01).map(_+1d)
  var dgamma = DenseVector.zeros[Double](xn)
  var eps = 1e-5
  var xhat = new Array[DenseVector[Double]](0)
  var xmu = new Array[DenseVector[Double]](0)
  var ivar = DenseVector[Double]()
  var sqrtvar = DenseVector[Double]()
  var varia = DenseVector[Double]()
  var opt = Opt.create("SGD",0.01)
  var allvaria = DenseVector.zeros[Double](xn)
  var allmu = DenseVector.zeros[Double](xn)

  override def forwards(x:Array[DenseVector[Double]]):Array[DenseVector[Double]]={
    var sum1 = DenseVector.zeros[Double](x(0).size)
    for(i<- 0 until x.size){
      sum1 = sum1 +:+ x(i)
    }
    val mu = (1d/x.size)*:*sum1

    xmu = x.map(_-mu)

    val sq = xmu.map{case a => a * a}

    var sum2 = DenseVector.zeros[Double](x(0).size)
    for(i<- 0 until x.size){
      sum2 = sum2 +:+ sq(i)
    }
    varia = (1d/x.size)*:*sum2

    sqrtvar = varia.map{case a => math.sqrt(a +eps)}

    ivar = sqrtvar.map{case a => 1d/a}

    xhat = xmu.map(_*:*ivar)

    val gammax = xhat.map(_*:*gamma)

    val out = gammax.map(_+:+beta)

    allmu = 0.9 * allmu +:+ 0.1 * mu
    allvaria = 0.9 *allvaria +:+ 0.1 * varia

    out
  }

  def forward(x:DenseVector[Double])={
    val allmux = x -:- allmu
    val xhattest = allmux /:/ (allvaria.map{case a => math.sqrt(a + eps)})
    val gammax = xhattest*:*gamma
    val out = gammax +:+ beta
    out
  }

  def backward(d:DenseVector[Double]) = {d}

  override def backwards(dout:Array[DenseVector[Double]]):Array[DenseVector[Double]]={
    for(i<- 0 until dout.size){
      dbeta = dbeta +:+ dout(i)
    }
    val dgammax = dout

    for(i<- 0 until dout.size){
      dgamma += dgammax(i)*:*xhat(i)
    }
    val dxhat = dgammax.map(_*:*gamma)

    val divar = DenseVector.zeros[Double](dout(0).size)
    for(i<- 0 until dout.size){
      divar += dxhat(i)*:*xmu(i)
    }
    val dxmu1 = dxhat.map(_*:*ivar)

    val dsqrtvar = (-1d/(sqrtvar*:*sqrtvar))*:*divar

    val dvar = 0.5 *:*(1d/(varia+:+eps).map(math.sqrt))*:*dsqrtvar

    val vec1 = (0 until dout.size).toArray.map{case a => dvar}
    val dsq = vec1.map(_*:*(1d/dout.size))

    val dxmu2 = xmu.zip(dsq).map{case (a,b)=> 2d*:*a*:*b}

    val dx1 = dxmu1.zip(dxmu2).map{case (a,b)=> a+:+b}
    val dmu = DenseVector.zeros[Double](dout(0).size)
    for(i<- 0 until dout.size){
      dmu -= dx1(i)
    }

    val vec2 = (0 until dout.size).toArray.map{case a => dmu}
    val dx2 = vec2.map(_*:*(1d/dout.size))

    val dx = dx1.zip(dx2).map{case (a,b)=> a+:+b}

    dx
  }

  def update(){
    val tmp1 = opt.update(Array(gamma),Array(dgamma))
    val tmp2 = opt.update(Array(beta),Array(dbeta))

    gamma = gamma - (tmp1(0))
    beta = beta - (tmp2(0))
    reset()
  }

  def reset(){
    dgamma = DenseVector.zeros[Double](xn)
    dbeta= DenseVector.zeros[Double](xn)
  }
  // def allsumreset(){
  //   var allsum1 = DenseVector.zeros[Double](xn)
  //   var allsum2 = DenseVector.zeros[Double](xn)
  //   var count = 0
  // }

  def save(fn:String){
    val fos = new java.io.FileOutputStream(fn,false)
    val osw = new java.io.OutputStreamWriter(fos,"UTF-8")
    val pw = new java.io.PrintWriter(osw)
    for(i <- 0 until gamma.size){
      pw.write(gamma(i).toString)
      if(i != gamma.size-1){
        pw.write(",")
      }
    }
    pw.write("\n")
    for(i <- 0 until beta.size){
      pw.write(beta(i).toString)
      if(i != beta.size-1){
        pw.write(",")
      }
    }
    pw.write("\n")
    pw.close()
  }

  def load(fn:String){
    val str = io.Source.fromFile(fn).getLines.toArray.map(_.split(",").map(_.toDouble))
    for(i <- 0 until gamma.size){
      gamma(i) = str(0)(i)
    }
    for(i <- 0 until beta.size){
      beta(i) = str(1)(i)
    }
  }

  def load(data: List[String]): List[String] = { data }
}