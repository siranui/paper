package pll
class LSTMfi(val xn: Int,
             val hn: Int,
             val dist: String,
             val n: Double,
             val u: String,
             val a: Double)
    extends Layer {
  import breeze.linalg._
  var wfx = DenseMatrix.zeros[Double](hn, xn)
  var wcx = DenseMatrix.zeros[Double](hn, xn)
  var wox = DenseMatrix.zeros[Double](hn, xn)
  var wfh = DenseMatrix.zeros[Double](hn, hn)
  var wch = DenseMatrix.zeros[Double](hn, hn)
  var woh = DenseMatrix.zeros[Double](hn, hn)
  var bf  = DenseVector.zeros[Double](hn)
  var bc  = DenseVector.zeros[Double](hn)
  var bo  = DenseVector.zeros[Double](hn)

  var dwfx = DenseMatrix.zeros[Double](hn, xn)
  var dwcx = DenseMatrix.zeros[Double](hn, xn)
  var dwox = DenseMatrix.zeros[Double](hn, xn)
  var dwfh = DenseMatrix.zeros[Double](hn, hn)
  var dwch = DenseMatrix.zeros[Double](hn, hn)
  var dwoh = DenseMatrix.zeros[Double](hn, hn)
  var ddbf = DenseVector.zeros[Double](hn)
  var ddbc = DenseVector.zeros[Double](hn)
  var ddbo = DenseVector.zeros[Double](hn)

  var mwfx = DenseMatrix.zeros[Double](hn, xn)
  var mwcx = DenseMatrix.zeros[Double](hn, xn)
  var mwox = DenseMatrix.zeros[Double](hn, xn)
  var mwfh = DenseMatrix.zeros[Double](hn, hn)
  var mwch = DenseMatrix.zeros[Double](hn, hn)
  var mwoh = DenseMatrix.zeros[Double](hn, hn)

  var vwfx = DenseMatrix.zeros[Double](hn, xn)
  var vwcx = DenseMatrix.zeros[Double](hn, xn)
  var vwox = DenseMatrix.zeros[Double](hn, xn)
  var vwfh = DenseMatrix.zeros[Double](hn, hn)
  var vwch = DenseMatrix.zeros[Double](hn, hn)
  var vwoh = DenseMatrix.zeros[Double](hn, hn)
  var t    = 0
  var beta = u.split(',')(1)

  if (dist == "Gaussian") {
    wfx = Gaussian(hn, xn, xn)
    wcx = Gaussian(hn, xn, xn)
    wox = Gaussian(hn, xn, xn)
    wfh = Gaussian(hn, hn, hn)
    wch = Gaussian(hn, hn, hn)
    woh = Gaussian(hn, hn, hn)
  }
  else if (dist == "Xavier") {
    wfx = Gaussian(hn, xn, xn)
    wcx = Gaussian(hn, xn, xn)
    wox = Gaussian(hn, xn, xn)
    wfh = Gaussian(hn, hn, hn)
    wch = Gaussian(hn, hn, hn)
    woh = Gaussian(hn, hn, hn)
  }
  else if (dist == "uniform") {
    wfx = Gaussian(hn, xn, xn)
    wcx = Gaussian(hn, xn, xn)
    wox = Gaussian(hn, xn, xn)
    wfh = Gaussian(hn, hn, hn)
    wch = Gaussian(hn, hn, hn)
    woh = Gaussian(hn, hn, hn)
  }
  else if (dist == "Xavier") {
    wfx = Xavier(hn, xn, xn)
    wcx = Xavier(hn, xn, xn)
    wox = Xavier(hn, xn, xn)
    wfh = Xavier(hn, hn, hn)
    wch = Xavier(hn, hn, hn)
    woh = Xavier(hn, hn, hn)
  }
  else if (dist == "He") {
    wfx = He(hn, xn, xn)
    wcx = He(hn, xn, xn)
    wox = He(hn, xn, xn)
    wfh = He(hn, hn, hn)
    wch = He(hn, hn, hn)
    woh = He(hn, hn, hn)
  }

  var opt = Opt.create(u, a)
  opt.register(Array(bf, bc, bo))
  opt.register(Array(wfx, wcx, wox, wfh, wch, woh))
  //opt.register(Array(wfh,wch,woh))

  var xl                       = List[DenseVector[Double]]()
  var fl                       = List[DenseVector[Double]]()
  var cl                       = List[DenseVector[Double]]()
  var ol                       = List[DenseVector[Double]]()
  var gl                       = List[DenseVector[Double]]()
  var Cl                       = List(DenseVector.zeros[Double](hn))
  var hl                       = List(DenseVector.zeros[Double](hn))
  var drt: DenseVector[Double] = null
  var dCt: DenseVector[Double] = null

  def sigmoid(x: DenseVector[Double]): DenseVector[Double] = {
    val a = (0d - x).map(math.exp)
    1d / (1d +:+ a)
    // x.map(a => 1d / (1d + math.exp(-a)))
  }

  def forward(x: DenseVector[Double]) = {
    drt = DenseVector.zeros[Double](hn)
    dCt = DenseVector.zeros[Double](hn)
    val ft = sigmoid(wfx * x + wfh * hl(0) + bf)
    val ct = (wcx * x + wch * hl(0) + bc).map(Math.tanh)
    val ot = sigmoid(wox * x + woh * hl(0) + bo)
    val Ct = (ft *:* Cl(0)) + ((1d - ft) *:* ct)
    val gt = Ct.map(Math.tanh)
    val ht = ot *:* gt

    xl = x :: xl
    fl = ft :: fl
    cl = ct :: cl
    ol = ot :: ol
    Cl = Ct :: Cl
    gl = gt :: gl
    hl = ht :: hl
    ht
  }

  def backward(dht: DenseVector[Double]) = {
    val dhr = dht + drt
    val dot = dhr *:* gl(0)
    val dg  = ol(0) *:* dhr
    val dbo = (ol(0) *:* (1d - ol(0))) *:* dot
    val dox = wox.t * dbo
    val dC  = (1d - (gl(0) *:* gl(0))) *:* dg + dCt
    val dc  = (1d - fl(0)) *:* dC
    val dbc = (1d - (cl(0) *:* cl(0))) *:* dc
    val dcx = wcx.t * dbc
    Cl = Cl.tail
    val df  = (dC *:* Cl(0)) - (dC *:* cl(0))
    val dbf = (fl(0) *:* (1d - fl(0))) *:* df
    val dfx = wfx.t * dbf
    dCt = fl(0) *:* dC
    drt = wfh.t * dbf + wch.t * dbc + woh.t * dbo

    hl = hl.tail
    dwfh += dbf * hl(0).t
    dwch += dbc * hl(0).t
    dwoh += dbo * hl(0).t
    dwfx += dbf * xl(0).t
    dwcx += dbc * xl(0).t
    dwox += dbo * xl(0).t
    ddbf += dbf
    ddbc += dbc
    ddbo += dbo

    xl = xl.tail
    fl = fl.tail
    cl = cl.tail
    ol = ol.tail
    gl = gl.tail

    val dxt = dfx + dcx + dox
    dxt
  }

  def update() {

    val dbs = opt.update(Array(bf, bc, bo), Array(ddbf, ddbc, ddbo))
    bf -= dbs(0)
    bc -= dbs(1)
    bo -= dbs(2)

    val dwxh =
      opt.update(Array(wfx, wcx, wox, wfh, wch, woh), Array(dwfx, dwcx, dwox, dwfh, dwch, dwoh))
    wfx -= dwxh(0)
    wcx -= dwxh(1)
    wox -= dwxh(2)
    wfh -= dwxh(3)
    wch -= dwxh(4)
    woh -= dwxh(5)

    dwfx = DenseMatrix.zeros[Double](hn, xn)
    dwcx = DenseMatrix.zeros[Double](hn, xn)
    dwox = DenseMatrix.zeros[Double](hn, xn)
    dwfh = DenseMatrix.zeros[Double](hn, hn)
    dwch = DenseMatrix.zeros[Double](hn, hn)
    dwoh = DenseMatrix.zeros[Double](hn, hn)
    ddbf = DenseVector.zeros[Double](hn)
    ddbc = DenseVector.zeros[Double](hn)
    ddbo = DenseVector.zeros[Double](hn)

    reset()
  }

  def reset() {
    /*
    opt.register(Array(bf,bc,bo))
    opt.register(Array(wfx,wcx,wox,wfh,wch,woh))
     */
    Cl = List(DenseVector.zeros[Double](hn))
    hl = List(DenseVector.zeros[Double](hn))
  }

  def save(filename: String) {}
  def load(filename: String) {}
  override def load(data: List[String]) = {
    data
  }
}
