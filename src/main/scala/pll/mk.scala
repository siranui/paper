package pll

object mk{

  def Graph(
    data: List[List[Double]],
    xlabel: String = "",
    ylabel: String = "",
    title: String = "",
    saveFileName: String = "result.png"
  ) = {
    import breeze.linalg._
    import breeze.plot._

    val f = Figure()
    val p = f.subplot(0)
    for(i <- 0 until data.size){
      val d = data(i).zipWithIndex
      i match{
        case 0 | 1 => p += plot(d.map(_._2.toDouble), d.map(_._1))
        case _ => p += plot(d.map(_._2.toDouble), d.map(_._1), '.')
      }
    }

    // 軸ラベル、タイトルの設定
    p.xlabel = xlabel; p.ylabel = ylabel; p.title = title
    f.saveas(saveFileName)
  }


}
