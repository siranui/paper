package pll


import breeze.linalg._

object Im2Col {
  type T = Double
  type DV = DenseVector[T]
  type DM = DenseMatrix[T]

  def im2col(x: Array[DV], fil_h: Int, fil_w: Int, ch: Int = 1, stride: Int = 1) = {
    val im: Array[Array[DV]] = x.map(b => utils.divideIntoN(b, ch))
    val in_w = math.sqrt(im(0)(0).size).toInt
    val images: Array[Array[DM]] = im.map( _.map( i => reshape(i, in_w, in_w).t ) )

    val out_w = utils.out_width(in_w, fil_w, stride)

    val col =
      ( for( image <- images ) yield {
        ( for( image_ch <- image) yield {
          ( for(i <- 0 until out_w; j <- 0 until out_w) yield {
            val m = image_ch(i*stride until i*stride+fil_h, j*stride until j*stride+fil_w).t
            m.reshape(fil_h*fil_w, 1)
          } ).reduce(DenseMatrix.horzcat(_, _))
        } ).reduce(DenseMatrix.vertcat(_, _))
      } ).reduce(DenseMatrix.horzcat(_, _))

    col
  }


  case class Shape4(B: Int, C: Int, H: Int, W: Int)

  def col2im(col: DM, x_shape: Shape4, fil_h: Int, fil_w: Int, stride: Int = 1): Array[Array[DM]] = {
    val batch = x_shape.B
    val ch = x_shape.C
    val in_h= x_shape.H
    val in_w = x_shape.W
    val out_h = utils.out_width(in_h, fil_h, stride)
    val out_w = utils.out_width(in_w, fil_w, stride)
    val fil_size = fil_h * fil_w

    val im = Array.fill(batch, ch)(DenseMatrix.zeros[Double](in_h, in_w))
    for {
      b <- 0 until batch
      c <- 0 until ch
      oh <- 0 until out_h
      ow <- 0 until out_w
    } {
      im(b)(c)(
        oh*stride until oh*stride+fil_h,
        ow*stride until ow*stride+fil_w
        ) += reshape(
          col(c*fil_size until c*fil_size+fil_size, b*out_h*out_w+oh*out_w+ow),
          fil_w,
          fil_h
        ).t
    }
    im
  }

}
