import breeze.linalg._

case class Conv(in_w:Int, out_w:Int, f_set:Int = 1, ch:Int = 1) {
  val fil_w = in_w - out_w + 1
  var F = DenseVector.rand(fil_w*fil_w)
  var Fs = Array.ofDim[DenseVector[Double]](f_set, ch)
  Fs = Fs.map(_.map(_ => F))

  // def forward(x: DenseVector[Double]) = {
  //   val o = F * x.t
  //   for(i <- 0 until out_w*out_w if(i%in_w<fil_w)){
  //     o(0, ::) := rotate.left(i,o(0,::).t).t
  //     o := rotate.up(1,o)
  //   }
  //   val oo = reshape(sum(o, Axis._0), in_w, in_w)
  //   oo(0 until out_w, 0 until out_w).t.toDenseVector
  // }

  def forward2(x: DenseVector[Double]) = {
    val tmp = (for(i <- 0 until x.length if(i%in_w<fil_w)) yield i).toArray
    // println(tmp.toList)
    val k = DenseVector.zeros[Double](x.length)
    for(i <- 0 until fil_w * fil_w){
      val t = F(i) * x
      println(tmp(i))
      k(0 until (k.length - tmp(i))) += t(tmp(i) until k.size)
      println(k)
    }
    k
  }

  def forward(x: DenseVector[Double]) = {
    val xs = pll.utils.divideIntoN(x, ch)
    val xmat = xs.map(_.toDenseMatrix).reduceLeft(DenseMatrix.vertcat(_,_))
    val Fmat = Fs.map(_.map(_.toDenseMatrix).reduce(DenseMatrix.vertcat(_,_))).reduce(DenseMatrix.vertcat(_,_)).reshape(ch, fil_w*fil_w*f_set/* = k^2*M */).t

    val res = Fmat * xmat

    val buf = DenseMatrix.zeros[Double](f_set, in_w*in_w)
    ((0 until fil_w*fil_w) zip (for(i <- 0 until in_w*in_w if(i % in_w < fil_w)) yield i)).foreach{ case (idx,num) =>
      buf +=  rotate.left(
        num,
        res(idx*f_set until idx*f_set+f_set,::)
      )
    }

    buf(*,::).map{ i =>
      val j = reshape(i.t, in_w, in_w)
      j(0 until out_w, 0 until out_w).t.toDenseVector
    }.t.toDenseVector
  }
}

object k2r {
  def check(
    in_w:Int = 3,
    out_w:Int = 2,
    f_set:Int = 2,
    ch:Int = 2
  )(implicit method: String = "normal") = {
    method match {
      case "k2r" | "kn2row" => kn2row(in_w, out_w, f_set, ch)
      case "normal" | _     => normal(in_w, out_w, f_set, ch)
    }
  }

  def kn2row(
    in_w:Int = 3,
    out_w:Int = 2,
    f_set:Int = 2,
    ch:Int = 2
  ) = {

    val fil_w = in_w - out_w + 1
    val filSize = fil_w * fil_w

    val x = convert(DenseVector.range(1,1+in_w*in_w*ch), Double)
    val f = convert(DenseVector.range(1,1+filSize), Double)
    var F = Array.ofDim[DenseVector[Double]](f_set, ch)
    F = F.map(_.map(_ => f))

    val kn2row = Conv(in_w,out_w,f_set,ch)
    kn2row.Fs = F

    kn2row.forward(x)
  }

  def normal(
    in_w:Int = 3,
    out_w:Int = 2,
    f_set:Int = 2,
    ch:Int = 2
  ) = {
    val fil_w = in_w - out_w + 1
    val filSize = fil_w * fil_w

    val x = convert(DenseVector.range(1,1+in_w*in_w*ch), Double)
    val f = convert(DenseVector.range(1,1+filSize), Double)
    var F = Array.ofDim[DenseVector[Double]](f_set, ch)
    F = F.map(_.map(_ => f))

    val conv = pll.Convolution(in_w, fil_w, f_set, ch)
    conv.F = F

    conv.forward(x)
  }
}
