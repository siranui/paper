package pll
import breeze.linalg._
trait Opt extends Load {
  def update(ps: Array[DenseMatrix[Double]],
             ds: Array[DenseMatrix[Double]]): Array[DenseMatrix[Double]];
  def update(ps: Array[DenseVector[Double]],
             ds: Array[DenseVector[Double]]): Array[DenseVector[Double]];
  def register(ps: Array[DenseMatrix[Double]]);
  def register(ps: Array[DenseVector[Double]]);
  def save(fn: String);
  def save_(pw: java.io.PrintWriter): java.io.PrintWriter = pw;
  // def load(fn: String);
  // def load(data: List[String]): List[String] = data;
}
object Opt {
  def create(name: String, lr: Double) = {
    val sp = name.split(",")
    if (sp(0) == "Momentum") {
      if (sp.size < 2) {
        if (lr == 0d)
          new Momentum()
        else
          new Momentum(lr)
      }
      else
        new Momentum(lr, sp(1).toDouble)
    }
    else if (sp(0) == "AdaGrad") {
      //println("AdaGrad")
      if (sp.size < 2) {
        if (lr == 0d)
          new AdaGrad()
        else
          new AdaGrad(lr)
      }
      else
        new AdaGrad(lr, sp(1).toDouble)
    }
    else if (sp(0) == "Adam") {
      //println("Adam")
      if (sp.size < 4) {
        if (lr == 0d)
          new Adam()
        else
          new Adam(lr)
      }
      else
        new Adam(lr, sp(1).toDouble, sp(2).toDouble, sp(3).toDouble)
    }
    else if (sp(0) == "RMSProp") {
      //println("RMSProp")
      if (sp.size < 3) {
        if (lr == 0d)
          new RMSProp()
        else
          new RMSProp(lr)
      }
      else
        new RMSProp(lr, sp(1).toDouble, sp(2).toDouble)
    }
    else
      new SGD(lr)
  }
}
class SGD(var lr: Double = 0.01) extends Opt {
  def register(ps: Array[DenseMatrix[Double]]) {}
  def register(ps: Array[DenseVector[Double]]) {}

  def update(ps: Array[DenseMatrix[Double]],
             ds: Array[DenseMatrix[Double]]): Array[DenseMatrix[Double]] = {
    var us = new Array[DenseMatrix[Double]](ps.size)
    for (i <- 0 until ps.size) {
      us(i) = lr * ds(i)
    }
    us
  }

  def update(ps: Array[DenseVector[Double]],
             ds: Array[DenseVector[Double]]): Array[DenseVector[Double]] = {
    var us = new Array[DenseVector[Double]](ps.size)
    for (i <- 0 until ps.size) {
      us(i) = lr * ds(i)
    }
    us
  }
  def save(fn: String) {}
  def load(fn: String) {}
}

class Momentum(var lr: Double = 0.01, var momentum: Double = 0.9) extends Opt {
  var vms = Array[DenseMatrix[Double]]()
  var vvs = Array[DenseVector[Double]]()

  def register(ps: Array[DenseMatrix[Double]]) {
    vms = new Array[DenseMatrix[Double]](ps.size)
    for (i <- 0 until vms.size)
      vms(i) = DenseMatrix.zeros[Double](ps(i).rows, ps(i).cols)
  }

  def register(ps: Array[DenseVector[Double]]) {
    vvs = new Array[DenseVector[Double]](ps.size)
    for (i <- 0 until vvs.size)
      vvs(i) = DenseVector.zeros[Double](ps(i).size)
  }

  def update(ps: Array[DenseMatrix[Double]],
             ds: Array[DenseMatrix[Double]]): Array[DenseMatrix[Double]] = {
    var us = new Array[DenseMatrix[Double]](ps.size)
    for (i <- 0 until ps.size) {
      vms(i) = (momentum * vms(i)) -:- (lr * ds(i))
      us(i) = -1d * vms(i)
    }
    us
  }

  def update(ps: Array[DenseVector[Double]],
             ds: Array[DenseVector[Double]]): Array[DenseVector[Double]] = {
    var us = new Array[DenseVector[Double]](ps.size)
    for (i <- 0 until ps.size) {
      vvs(i) = (momentum * vvs(i)) -:- (lr * ds(i))
      us(i) = -1d * vvs(i)
    }
    us
  }
  def save(fn: String) {}
  def load(fn: String) {}
}

class AdaGrad(var lr: Double = 0.01, var epsilon: Double = 1e-8) extends Opt {
  var hms = Array[DenseMatrix[Double]]()
  var hvs = Array[DenseVector[Double]]()

  def register(ps: Array[DenseMatrix[Double]]) {
    hms = new Array[DenseMatrix[Double]](ps.size)
    for (i <- 0 until hms.size)
      hms(i) = DenseMatrix.zeros[Double](ps(i).rows, ps(i).cols)
  }

  def register(ps: Array[DenseVector[Double]]) {
    hvs = new Array[DenseVector[Double]](ps.size)
    for (i <- 0 until hvs.size)
      hvs(i) = DenseVector.zeros[Double](ps(i).size)
  }

  def update(ps: Array[DenseMatrix[Double]],
             ds: Array[DenseMatrix[Double]]): Array[DenseMatrix[Double]] = {
    var us = new Array[DenseMatrix[Double]](ps.size)
    for (i <- 0 until ps.size) {
      hms(i) = hms(i) + ds(i) *:* ds(i)
      us(i) = lr * ds(i) /:/ (hms(i).map(Math.sqrt) + epsilon)
    }
    us
  }

  def update(ps: Array[DenseVector[Double]],
             ds: Array[DenseVector[Double]]): Array[DenseVector[Double]] = {
    var us = new Array[DenseVector[Double]](ps.size)
    for (i <- 0 until ps.size) {
      hvs(i) = hvs(i) + ds(i) *:* ds(i)
      us(i) = lr * ds(i) /:/ (hvs(i).map(Math.sqrt) + epsilon)
    }
    us
  }
  def save(fn: String) {}
  def load(fn: String) {}
}

class Adam(val lr: Double = 0.001,
           val beta1: Double = 0.9,
           val beta2: Double = 0.999,
           val epsilon: Double = 1e-8)
    extends Opt {
  var vms = Array[DenseMatrix[Double]]()
  var hms = Array[DenseMatrix[Double]]()
  var vvs = Array[DenseVector[Double]]()
  var hvs = Array[DenseVector[Double]]()
  var tm  = 0
  var tv  = 0

  def register(ps: Array[DenseMatrix[Double]]) {
    hms = new Array[DenseMatrix[Double]](ps.size)
    vms = new Array[DenseMatrix[Double]](ps.size)
    for (i <- 0 until hms.size) {
      hms(i) = DenseMatrix.zeros[Double](ps(i).rows, ps(i).cols)
      vms(i) = DenseMatrix.zeros[Double](ps(i).rows, ps(i).cols)
    }
  }

  def register(ps: Array[DenseVector[Double]]) {
    hvs = new Array[DenseVector[Double]](ps.size)
    vvs = new Array[DenseVector[Double]](ps.size)
    for (i <- 0 until hvs.size) {
      hvs(i) = DenseVector.zeros[Double](ps(i).size)
      vvs(i) = DenseVector.zeros[Double](ps(i).size)
    }
  }

  def update(ps: Array[DenseMatrix[Double]],
             ds: Array[DenseMatrix[Double]]): Array[DenseMatrix[Double]] = {
    tm += 1
    val t  = tm
    var us = new Array[DenseMatrix[Double]](ps.size)
    for (i <- 0 until ps.size) {
      vms(i) = beta1 * vms(i) + (1d - beta1) * ds(i)
      hms(i) = beta2 * hms(i) + (1d - beta2) * (ds(i) *:* ds(i))
      val vvms = vms(i) / (1d - Math.pow(beta1, t))
      val hhms = hms(i) / (1d - Math.pow(beta2, t))
      us(i) = lr * vvms /:/ (hhms.map(Math.sqrt) + epsilon)
    }
    us
  }

  def update(ps: Array[DenseVector[Double]],
             ds: Array[DenseVector[Double]]): Array[DenseVector[Double]] = {
    tv += 1
    val t  = tv
    var us = new Array[DenseVector[Double]](ps.size)
    for (i <- 0 until ps.size) {
      vvs(i) = beta1 * vvs(i) + (1d - beta1) * ds(i)
      hvs(i) = beta2 * hvs(i) + (1d - beta2) * (ds(i) *:* ds(i))
      val vvvs = vvs(i) / (1d - Math.pow(beta1, t))
      val hhvs = hvs(i) / (1d - Math.pow(beta2, t))
      us(i) = lr * vvvs /:/ (hhvs.map(Math.sqrt) + epsilon)
    }
    us
  }
  def save(fn: String) {
    val fos = new java.io.FileOutputStream(fn + "-opt.txt", false)
    val osw = new java.io.OutputStreamWriter(fos, "UTF-8")
    val pw  = new java.io.PrintWriter(osw)
    for (k <- 0 until hms.size) {
      for (i <- 0 until hms(k).rows) {
        for (j <- 0 until hms(k).cols) {
          pw.write(hms(k)(i, j).toString)
          pw.write(",")
        }
      }
      pw.write("\n")
    }
    for (k <- 0 until vms.size) {
      for (i <- 0 until vms(k).rows) {
        for (j <- 0 until vms(k).cols) {
          pw.write(vms(k)(i, j).toString)
          pw.write(",")
        }
      }
      pw.write("\n")
    }
    for (k <- 0 until hvs.size) {
      for (i <- 0 until hvs(k).size) {
        pw.write(hvs(k)(i).toString)
        pw.write(",")
      }
      pw.write("\n")
    }
    for (k <- 0 until vvs.size) {
      for (i <- 0 until vvs(k).size) {
        pw.write(vvs(k)(i).toString)
        pw.write(",")
      }
      pw.write("\n")
    }
    pw.write(tm.toString)
    pw.write("\n")
    pw.write(tv.toString)
    pw.write("\n")
    pw.close()
  }

  override def save_(pw: java.io.PrintWriter): java.io.PrintWriter = {
    for (k <- 0 until hms.size) {
      for (i <- 0 until hms(k).rows) {
        for (j <- 0 until hms(k).cols) {
          pw.write(hms(k)(i, j).toString)
          pw.write(",")
        }
      }
      pw.write("\n")
    }
    for (k <- 0 until vms.size) {
      for (i <- 0 until vms(k).rows) {
        for (j <- 0 until vms(k).cols) {
          pw.write(vms(k)(i, j).toString)
          pw.write(",")
        }
      }
      pw.write("\n")
    }
    for (k <- 0 until hvs.size) {
      for (i <- 0 until hvs(k).size) {
        pw.write(hvs(k)(i).toString)
        pw.write(",")
      }
      pw.write("\n")
    }
    for (k <- 0 until vvs.size) {
      for (i <- 0 until vvs(k).size) {
        pw.write(vvs(k)(i).toString)
        pw.write(",")
      }
      pw.write("\n")
    }
    pw.write(tm.toString)
    pw.write("\n")
    pw.write(tv.toString)
    pw.write("\n")
    pw
  }

  def load(fn: String) {
    val str = io.Source.fromFile(fn + "-opt.txt").getLines.toArray.map(_.split(",").map(_.toDouble))
    var num = 0
    for (k <- 0 until hms.size) {
      for (i <- 0 until hms(k).rows) {
        for (j <- 0 until hms(k).cols) {
          hms(k)(i, j) = str(num)(hms(k).cols * i + j)
        }
      }
      num += 1
    }
    for (k <- 0 until vms.size) {
      for (i <- 0 until vms(k).rows) {
        for (j <- 0 until vms(k).cols) {
          vms(k)(i, j) = str(num)(vms(k).cols * i + j)
        }
      }
      num += 1
    }
    for (k <- 0 until hvs.size) {
      for (i <- 0 until hvs(k).size) {
        hvs(k)(i) = str(num)(i)
      }
      num += 1
    }
    for (k <- 0 until vvs.size) {
      for (i <- 0 until vvs(k).size) {
        vvs(k)(i) = str(num)(i)
      }
      num += 1
    }
    tm = str(num)(0).toInt
    num += 1
    tv = str(num)(0).toInt
  }

  override def load(data: List[String]): List[String] = {
    val str = data.map(_.split(",").map(_.toDouble))
    pll.log.debug(s"adam-load:")
    var num = 0
    for (k <- 0 until hms.size) {
      pll.log.debug(s"\thms($k):${hms(k).size}, loaded: ${str(num).size}")
      for (i <- 0 until hms(k).rows) {
        for (j <- 0 until hms(k).cols) {
          hms(k)(i, j) = str(num)(hms(k).cols * i + j).toDouble
        }
      }
      num += 1
    }
    for (k <- 0 until vms.size) {
      pll.log.debug(s"\tvms($k):${vms(k).size}, loaded: ${str(num).size}")
      for (i <- 0 until vms(k).rows) {
        for (j <- 0 until vms(k).cols) {
          vms(k)(i, j) = str(num)(vms(k).cols * i + j)
        }
      }
      num += 1
    }
    for (k <- 0 until hvs.size) {
      pll.log.debug(s"\thvs($k):${hvs(k).size}, loaded: ${str(num).size}")
      for (i <- 0 until hvs(k).size) {
        hvs(k)(i) = str(num)(i)
      }
      num += 1
    }
    for (k <- 0 until vvs.size) {
      pll.log.debug(s"\tvvs($k):${vvs(k).size}, loaded: ${str(num).size}")
      for (i <- 0 until vvs(k).size) {
        vvs(k)(i) = str(num)(i)
      }
      num += 1
    }
    pll.log.debug(s"\ttm:${1}, loaded: ${str(num).size}")
    tm = str(num)(0).toInt
    num += 1

    pll.log.debug(s"\ttv:${1}, loaded: ${str(num).size}")
    tv = str(num)(0).toInt
    data.drop(num + 1)
  }

  override def load_version_iterator(data_iter: scala.io.BufferedSource): Unit = {
    for (k <- 0 until hms.size) {
      for (i <- 0 until hms(k).rows) {
        for (j <- 0 until hms(k).cols) {
          hms(k)(i, j) = get_value(data_iter)
        }
      }
    }
    for (k <- 0 until vms.size) {
      for (i <- 0 until vms(k).rows) {
        for (j <- 0 until vms(k).cols) {
          vms(k)(i, j) = get_value(data_iter)
        }
      }
    }
    for (k <- 0 until hvs.size) {
      for (i <- 0 until hvs(k).size) {
        hvs(k)(i) = get_value(data_iter)
      }
    }
    for (k <- 0 until vvs.size) {
      for (i <- 0 until vvs(k).size) {
        vvs(k)(i) = get_value(data_iter)
      }
    }
    tm = get_value(data_iter).toInt

    tv = get_value(data_iter).toInt
  }
}

class RMSProp(val lr: Double = 0.001, val beta: Double = 0.9, val epsilon: Double = 1e-8)
    extends Opt {
  var hms = Array[DenseMatrix[Double]]()
  var hvs = Array[DenseVector[Double]]()

  def register(ps: Array[DenseMatrix[Double]]) {
    hms = new Array[DenseMatrix[Double]](ps.size)
    for (i <- 0 until hms.size) {
      hms(i) = DenseMatrix.zeros[Double](ps(i).rows, ps(i).cols)
    }
  }

  def register(ps: Array[DenseVector[Double]]) {
    hvs = new Array[DenseVector[Double]](ps.size)
    for (i <- 0 until hvs.size) {
      hvs(i) = DenseVector.zeros[Double](ps(i).size)
    }
  }

  def update(ps: Array[DenseMatrix[Double]],
             ds: Array[DenseMatrix[Double]]): Array[DenseMatrix[Double]] = {
    var us = new Array[DenseMatrix[Double]](ps.size)
    for (i <- 0 until ps.size) {
      hms(i) = beta * hms(i) + (1d - beta) * (ds(i) *:* ds(i))
      us(i) = lr * ds(i) /:/ (hms(i).map(Math.sqrt) + epsilon)
    }
    us
  }

  def update(ps: Array[DenseVector[Double]],
             ds: Array[DenseVector[Double]]): Array[DenseVector[Double]] = {
    var us = new Array[DenseVector[Double]](ps.size)
    for (i <- 0 until ps.size) {
      hvs(i) = beta * hvs(i) + (1d - beta) * (ds(i) *:* ds(i))
      us(i) = lr * ds(i) /:/ (hvs(i).map(Math.sqrt) + epsilon)
    }
    us
  }
  def save(fn: String) {}
  def load(fn: String) {}
}
