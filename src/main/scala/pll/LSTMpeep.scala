package pll
class LSTMpeep(val xn:Int,val hn:Int,val dist:String,var n:Double,val u:String,val a:Double) extends Layer{
  import breeze.linalg._
  var clist = List[DenseVector[Double]](DenseVector.zeros[Double](hn))
  var hlist = List[DenseVector[Double]](DenseVector.zeros[Double](hn))
  var xlist = List[DenseVector[Double]]()
  var flist = List[DenseVector[Double]]()
  var ilist = List[DenseVector[Double]]()
  var c1list = List[DenseVector[Double]]()
  var olist = List[DenseVector[Double]]()
  var wfh = DenseMatrix.zeros[Double](hn,hn)
  var wfx = DenseMatrix.zeros[Double](hn,xn)
  var wfc = DenseMatrix.zeros[Double](hn,hn)
  var woh = DenseMatrix.zeros[Double](hn,hn)
  var wox = DenseMatrix.zeros[Double](hn,xn)
  var woc = DenseMatrix.zeros[Double](hn,hn)
  var wix = DenseMatrix.zeros[Double](hn,xn)
  var wih = DenseMatrix.zeros[Double](hn,hn)
  var wic = DenseMatrix.zeros[Double](hn,hn)
  var wch = DenseMatrix.zeros[Double](hn,hn)
  var wcx = DenseMatrix.zeros[Double](hn,xn)
  var bf = DenseVector.ones[Double](hn)
  var bi = DenseVector.zeros[Double](hn)
  var bc = DenseVector.zeros[Double](hn)
  var bo = DenseVector.zeros[Double](hn)
  var dc = DenseVector.zeros[Double](hn)
  var dr = DenseVector.zeros[Double](hn)

  var dO = DenseVector.zeros[Double](hn)
  var dbo = DenseVector.zeros[Double](hn)
  var dwox = DenseMatrix.zeros[Double](hn,xn)
  var dwoh = DenseMatrix.zeros[Double](hn,hn)
  var dwoc = DenseMatrix.zeros[Double](hn,hn)

  var du = DenseVector.zeros[Double](hn)

  var dc1 = DenseVector.zeros[Double](hn)
  var dbc = DenseVector.zeros[Double](hn)
  var dwcx = DenseMatrix.zeros[Double](hn,xn)
  var dwch = DenseMatrix.zeros[Double](hn,hn)

  var di = DenseVector.zeros[Double](hn)
  var dbi = DenseVector.zeros[Double](hn)
  var dwix = DenseMatrix.zeros[Double](hn,xn)
  var dwih = DenseMatrix.zeros[Double](hn,hn)
  var dwic = DenseMatrix.zeros[Double](hn,hn)

  var df = DenseVector.zeros[Double](hn)
  var dbf = DenseVector.zeros[Double](hn)
  var dwfx = DenseMatrix.zeros[Double](hn,xn)
  var dwfh = DenseMatrix.zeros[Double](hn,hn)
  var dwfc = DenseMatrix.zeros[Double](hn,hn)

  var f = DenseVector.zeros[Double](hn)
  var i = DenseVector.zeros[Double](hn)
  var c1 = DenseVector.zeros[Double](hn)
  var o = DenseVector.zeros[Double](hn)


  if(dist == "Gaussian"){
    wfx = Gaussian(hn,xn,xn)
    wix = Gaussian(hn,xn,xn)
    wcx = Gaussian(hn,xn,xn)
    woc = Gaussian(hn,hn,hn)
    wfh = Gaussian(hn,hn,hn)
    wfc = Gaussian(hn,hn,hn)
    woh = Gaussian(hn,hn,hn)
    woc = Gaussian(hn,hn,hn)
    wih = Gaussian(hn,hn,hn)
    wic = Gaussian(hn,hn,hn)
    wch = Gaussian(hn,hn,hn)
  }else if(dist == "Uniform"){
    wfx = Uniform(hn,xn,xn)
    wix = Uniform(hn,xn,xn)
    wcx = Uniform(hn,xn,xn)
    woc = Uniform(hn,hn,hn)
    wfh = Uniform(hn,hn,hn)
    wfc = Uniform(hn,hn,hn)
    woh = Uniform(hn,hn,hn)
    woc = Uniform(hn,hn,hn)
    wih = Uniform(hn,hn,hn)
    wic = Uniform(hn,hn,hn)
    wch = Uniform(hn,hn,hn)
  }else if(dist == "Xavier"){
    wfx = Xavier(hn,xn,xn)
    wix = Xavier(hn,xn,xn)
    wcx = Xavier(hn,xn,xn)
    woc = Xavier(hn,hn,hn)
    wfh = Xavier(hn,hn,hn)
    wfc = Xavier(hn,hn,hn)
    woh = Xavier(hn,hn,hn)
    woc = Xavier(hn,hn,hn)
    wih = Xavier(hn,hn,hn)
    wic = Xavier(hn,hn,hn)
    wch = Xavier(hn,hn,hn)
  }else if(dist == "He"){
    wfx = He(hn,xn,hn)
    wix = He(hn,xn,hn)
    wcx = He(hn,xn,hn)
    woc = He(hn,hn,hn)
    wfh = He(hn,hn,hn)
    wfc = He(hn,hn,hn)
    woh = He(hn,hn,hn)
    woc = He(hn,hn,hn)
    wih = He(hn,hn,hn)
    wic = He(hn,hn,hn)
    wch = He(hn,hn,hn)
  }

  var opt = Opt.create(u,a)
  opt.register(Array(bf,bc,bo))
  opt.register(Array(wfx,wix,wcx,wox,wfh,wih,wch,woh))


  def sigmoid(x:DenseVector[Double]):DenseVector[Double] = {
    val a = (0d-x).map(math.exp)
    1d/(1d+:+a)
    // x.map(a => 1d / (1d + math.exp(-a)))
  }

  def forward(x:DenseVector[Double]) = {
    xlist = x :: xlist
    //println(hlist(0) , Clist(0) , xlist(0))
    f = sigmoid(wfh*hlist(0) + wfx*x + wfc*clist(0) + bf)
    i = sigmoid(wih*hlist(0) + wix*x + wic*clist(0) + bi)
    c1 = (wcx*x + wch*hlist(0)+bc).map(math.tanh)
    val c = (f*:*clist(0)) + (i*:*c1)
    clist = c :: clist
    o = sigmoid(woh*hlist(0) + wox*x + woc*clist(0) +  bo)

    val h = c.map(math.tanh)*:*o

    hlist = h :: hlist
    flist = f :: flist
    ilist = i :: ilist
    olist = o :: olist
    c1list = c1 :: c1list
    h
  }

  def backward(dh:DenseVector[Double]) = {
    val z = (clist(0)).map(math.tanh)
    val f = flist(0)
    val i = ilist(0)
    val o = olist(0)
    val c1 = c1list(0)

    dO = (dh+:+dr)*:*z
    val dbo1 = (o*:*(1d-o)*:*dO)
    dbo += dbo1

    dwox += dbo1*xlist(0).t
    dwoh += dbo1*hlist(1).t
    dwoc += dbo1*clist(0).t

    du = woc.t*dbo1 + dc + ((1d-(z*:*z))*:*(dh+:+dr)*:*o)

    dc1 = du*:*i
    val dbc1 = ((1d-(c1*:*c1))*:*dc1)
    dbc += dbc1
    dwcx += dbc1*xlist(0).t
    dwch += dbc1*hlist(1).t

    clist = clist.tail

    di = du*:*c1
    val dbi1 = (i*:*(1d-i)*:*di)
    dbi += dbi1
    dwix += dbi1*xlist(0).t
    dwih += dbi1*hlist(1).t
    dwic += dbi1*clist(0).t

    df = du*:*clist(0)
    val dbf1 = (f*:*(1d-f)*:*df)
    dbf += dbf1
    dwfx += dbf1*xlist(0).t
    dwfh += dbf1*hlist(1).t
    dwfc += dbf1*clist(0).t

    xlist = xlist.tail
    hlist = hlist.tail
    flist = flist.tail
    ilist = ilist.tail
    olist = olist.tail
    c1list = c1list.tail

    dc = (du*:*f) + wic.t*dbi1 + wfc.t*dbf1
    dr = wfh.t*dbf1 + wih.t*dbi1 + wch.t*dbc1 + woh.t*dbo1
    val dx = wfx.t*dbf1 + wix.t*dbi1 + wcx.t*dbc1 + wox.t*dbo1
    dx
  }

  def update() = {

/*
    woc -= (a*:*dwoc)
    wox -= (a*:*dwox)
    woh -= (a*:*dwoh)
    bo -= (a*:*dbo)

    wcx -= (a*:*dwcx)
    wch -= (a*:*dwch)
    bc -= (a*:*dbc)

    wic -= (a*:*dwic)
    wix -= (a*:*dwix)
    wih -= (a*:*dwih)
    bi -= (a*:*dbi)

    wfc -= (a*:*dwfc)
    wfx -= (a*:*dwfx)
    wfh -= (a*:*dwfh)
    bf -= (a*:*dbf)
 */
    val dbs = opt.update(Array(bf,bi,bc,bo),Array(dbf,dbi,dbc,dbo))
    bf -= dbs(0)
    bi -= dbs(1)
    bc -= dbs(2)
    bo -= dbs(3)

    val dwxh = opt.update(Array(wfx,wix,wcx,wox,wfh,wih,wch,woh),Array(dwfx,dwix,dwcx,dwox,dwfh,dwih,dwch,dwoh))
    wfx -= dwxh(0)
    wix -= dwxh(1)
    wcx -= dwxh(2)
    wox -= dwxh(3)
    wfh -= dwxh(4)
    wih -= dwxh(5)
    wch -= dwxh(6)
    woh -= dwxh(7)

    dr = DenseVector.zeros[Double](hn)
    dc = DenseVector.zeros[Double](hn)
    reset()

    dwox = DenseMatrix.zeros[Double](hn,xn)
    dwoh = DenseMatrix.zeros[Double](hn,hn)
    dwoc = DenseMatrix.zeros[Double](hn,hn)
    dwcx = DenseMatrix.zeros[Double](hn,xn)
    dwch = DenseMatrix.zeros[Double](hn,hn)

    dwix = DenseMatrix.zeros[Double](hn,xn)
    dwih = DenseMatrix.zeros[Double](hn,hn)
    dwic = DenseMatrix.zeros[Double](hn,hn)

    dwfx = DenseMatrix.zeros[Double](hn,xn)
    dwfh = DenseMatrix.zeros[Double](hn,hn)
    dwfc = DenseMatrix.zeros[Double](hn,hn)

    dbf = DenseVector.zeros[Double](hn)
    dbi = DenseVector.zeros[Double](hn)
    dbc = DenseVector.zeros[Double](hn)
    dbo = DenseVector.zeros[Double](hn)
  }


  def reset() = {
    clist = List[DenseVector[Double]](DenseVector.zeros[Double](hn))
    hlist = List[DenseVector[Double]](DenseVector.zeros[Double](hn))
    xlist = List[DenseVector[Double]]()
    flist = List[DenseVector[Double]]()
    ilist = List[DenseVector[Double]]()
    olist = List[DenseVector[Double]]()
    c1list = List[DenseVector[Double]]()
  }

  def save(filename:String){

  }
  def load(filename:String){

  }
  def load(data: List[String]) = {
    data
  }

}
