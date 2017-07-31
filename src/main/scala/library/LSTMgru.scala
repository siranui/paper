package pll
import breeze.linalg._

class GRU(val xn:Int,val hn:Int,val dist:String,var n:Double,val u:String,val a:Double) extends Layer {
  //xn = ベクトルの要素数
  //hn = 中間層のノード数

  //前の入力をとっておく
  var xs = List[DenseVector[Double]]()
  var wx = xavier(hn,xn)
  var wh = xavier(hn,hn)
  var wrx = xavier(hn,xn)
  var wrh = xavier(hn,hn)
  var wzx = xavier(hn,xn)
  var wzh = xavier(hn,hn)
  //重みの更新量
  var wxsum = DenseMatrix.zeros[Double](wx.rows,wx.cols)
  var whsum = DenseMatrix.zeros[Double](wh.rows,wh.cols)
  var wzxsum = DenseMatrix.zeros[Double](wzx.rows,wzx.cols)
  var wzhsum = DenseMatrix.zeros[Double](wzh.rows,wzh.cols)
  var wrxsum = DenseMatrix.zeros[Double](wrx.rows,wrx.cols)
  var wrhsum = DenseMatrix.zeros[Double](wrh.rows,wrh.cols)

  if(dist == "Gaussian"){
    wx = Gaussian(hn,xn,xn)
    wrx = Gaussian(hn,xn,xn)
    wzx = Gaussian(hn,xn,xn)
    wh = Gaussian(hn,hn,hn)
    wrh = Gaussian(hn,hn,hn)
    wzh = Gaussian(hn,hn,hn)
  }else if(dist == "Uniform"){
    wx = Uniform(hn,xn,xn)
    wrx = Uniform(hn,xn,xn)
    wzx = Uniform(hn,xn,xn)
    wh = Uniform(hn,hn,hn)
    wrh = Uniform(hn,hn,hn)
    wzh = Uniform(hn,hn,hn)
  }else if(dist == "Xavier"){
    wx = Xavier(hn,xn,xn)
    wrx = Xavier(hn,xn,xn)
    wzx = Xavier(hn,xn,xn)
    wh = Xavier(hn,hn,hn)
    wrh = Xavier(hn,hn,hn)
    wzh = Xavier(hn,hn,hn)
  }else if(dist == "He"){
    wx = He(hn,xn,xn)
    wrx = He(hn,xn,xn)
    wzx = He(hn,xn,xn)
    wh = He(hn,hn,hn)
    wrh = He(hn,hn,hn)
    wzh = He(hn,hn,hn)
  }

  var opt = Opt.create(u,a)
  opt.register(Array(wx,wzx,wrx,wh,wzh,wrh))

  val zerovec = DenseVector.zeros[Double](hn)
  var h : List[DenseVector[Double]] = List(zerovec)
  var l = DenseVector.zeros[Double](hn)
  var r = List[DenseVector[Double]]()
  var z = List[DenseVector[Double]]()
  var h1 = List[DenseVector[Double]]()

  def makeh(hh:DenseVector[Double]){
    h = List(hh)
  }
  def makel(ll:DenseVector[Double]){
    l = ll
  }

  def forward(x:DenseVector[Double])={
    r = sigmoid(wrx*x+wrh*h(0)) :: r
    z = sigmoid(wzx*x+wzh*h(0)) :: z
    h1 = (wx*x+(wh*(h(0)*:*r(0)))).map(math.tanh) :: h1
    h = (z(0)*:*h1(0)) + (h(0)*:*f(z(0))) :: h
    xs = x :: xs
    h(0)
  }
  def backward(dht:DenseVector[Double]) ={
    val dh = dht + l
    val dh1 = dh*:*z(0)
    val dx1 = dh1 *:* (1d - (h1(0) *:* h1(0)))
    wxsum += dx1 * xs(0).t
    whsum += (dx1 *:* h(1)) * r(0).t
    val dz = -1d *(dh *:* h(1)) + (dh *:* h1(0))
    val dx2 = dz *:* (z(0) *:* (1d - z(0)))
    wzxsum += dx2 * xs(0).t
    wzhsum += dx2 * h(1).t
    val dr = (wh * dx1) *:* h(1)
    val dx3 = dr *:* (r(0) *:* (1d - r(0)))
    wrhsum += dx3 * h(1).t
    wrxsum += dx3 * xs(0).t
    l = ((wh * dx1) *:* r(0)) + wzh * dx2 + wrh * dx3 + (f(z(0)) *:* dh)
    r = r.tail
    z = z.tail
    h = h.tail
    h1 = h1.tail
    xs = xs.tail
    wx.t * dx1 + wzx.t * dx2 + wrx.t * dx3
  }
  def update(){
/*

    wx = wx - (wxsum * a)
    wh = wh - (whsum * a)
    wzx = wzx - (wzxsum * a)
    wzh = wzh - (wzhsum * a)
    wrx = wrx - (wrxsum * a)
    wrh = wrh - (wrhsum * a)
 */


    val dwxh = opt.update(Array(wx,wzx,wrx,wh,wzh,wrh),Array(wxsum,wzxsum,wrxsum,whsum,wzhsum,wrhsum))
    wx -= dwxh(0)
    wzx -= dwxh(1)
    wrx -= dwxh(2)
    wh -= dwxh(3)
    wzh -= dwxh(4)
    wrh -= dwxh(5)

    wxsum = DenseMatrix.zeros[Double](hn,xn)
    whsum = DenseMatrix.zeros[Double](hn,hn)
    wzxsum = DenseMatrix.zeros[Double](hn,xn)
    wzhsum = DenseMatrix.zeros[Double](hn,hn)
    wrxsum = DenseMatrix.zeros[Double](hn,xn)
    wrhsum = DenseMatrix.zeros[Double](hn,hn)
    }
  def reset(){
    l = DenseVector.zeros[Double](hn)
    h = List(zerovec)
    xs = List[DenseVector[Double]]()
    r = List[DenseVector[Double]]()
    z = List[DenseVector[Double]]()
    h1 = List[DenseVector[Double]]()
  }
  //ザビエの初期値
  def xavier(r:Int,c:Int) ={
    val x = DenseMatrix.zeros[Double](r,c)
    for(i <- 0 until r){
      for(j <- 0 until c){
        x(i,j) = rand.nextGaussian*(1/math.sqrt(c))
      }
    }
    x
  }

  def f(x:DenseVector[Double])= {
    val one = DenseVector.ones[Double](x.size)
    one - x
  }

  def sigmoid(x:DenseVector[Double])= {
    x.map(a => 1/(1+math.exp(-a)))
  }

  def save(filename:String){

  }
  def load(data: List[String]) = {
    data
  }
}
