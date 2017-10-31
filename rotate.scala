object rotate {
  import breeze.linalg._
  def left[T](n:Int, s:Seq[T])  = s.drop(n%s.size) ++ s.take(n%s.size)
  def right[T](n:Int, s:Seq[T]) = s.takeRight(n%s.size) ++ s.dropRight(n%s.size)

  def left(n:Int, v:DenseVector[Double])  = DenseVector.vertcat(v(n%v.size until v.size), v(0 until n%v.size))
  def right(n:Int, v:DenseVector[Double]) = DenseVector.vertcat(v(v.size - n%v.size until v.size), v(0 until v.size - n%v.size))

  def left(n:Int, m:DenseMatrix[Double])  = DenseMatrix.horzcat(m(::, n%m.cols until m.cols), m(::, 0 until n%m.cols))
  def right(n:Int, m:DenseMatrix[Double]) = DenseMatrix.horzcat(m(::, m.cols - n%m.cols until m.cols), m(::, 0 until m.cols - n%m.cols))
  def up(n:Int, m:DenseMatrix[Double])    = DenseMatrix.vertcat(m(n%m.rows until m.rows, ::), m(0 until n%m.rows, ::))
  def down(n:Int, m:DenseMatrix[Double])  = DenseMatrix.vertcat(m(m.rows - n%m.rows until m.rows, ::), m(0 until m.rows - n%m.rows, ::))
}
