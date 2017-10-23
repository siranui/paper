package com.lightbend.akka.sample

import breeze.linalg._
import pll._

object MNIST {

  val INPUT = 784
  val HIDDEN = 100
  val OUTPUT = 10

  val datasize = 500
  val testsize = 100

  var doShuffle = false

  val train_d = utils.read("data/mnist/train-d.txt",datasize)
  val train_t = utils.read("data/mnist/train-t.txt",1,1)
  val test_d = utils.read("data/mnist/test-d.txt",testsize)
  val test_t = utils.read("data/mnist/test-t.txt",1,1)

  val epoch = 30


  def main(args: Array[String]) {
    import breeze.plot._
    import pll.actors._
    import pll.actors.Monitor._
    import akka.actor.ActorSystem

    val actorSystem = ActorSystem("actor-system")
    val hist = Figure()
    val err_rate = Figure()
    val histActor = actorSystem.actorOf(Monitor.props(hist), "histgram-actor")
    val errRateActor = actorSystem.actorOf(Monitor.props(err_rate), "error-rate-actor")


    // シャッフルするかどうか
    args(0) match {
      case "true"      => doShuffle = true
      case "false" | _ => doShuffle = false
    }

    val rand = new util.Random(0)

    val net = new Network()

    net.add(new Affine(INPUT, HIDDEN, "He", 1, "SGD", 0.01)).
    add(new ReLU()).
    add(new Affine(HIDDEN, OUTPUT, "Xavier", 1, "SGD", 0.01)).
    add(new SoftMax())

    var E_list: List[Double] = Nil
    var tE_list: List[Double] = Nil

    for(i <- 0 until epoch){
      var dataset = train_d zip train_t(0).toArray
      if(doShuffle){
        val tmp = rand.shuffle(List.range(0, datasize)).toArray
        dataset = tmp.map(i => dataset(i))
      }

      val testset = test_d zip test_t(0).toArray

      var E = 0d
      var tE = 0d

      var acc = 0d
      var tacc = 0d

      val mixMat = DenseMatrix.zeros[Int](10,10)
      val tmixMat = DenseMatrix.zeros[Int](10,10)

      var ii = 0
      // training
      for((data, tag) <- dataset){

        val y = if(i%5 == 0 && ii == 10){
          var predict_value = data
          var vecs: List[DenseVector[Double]] = Nil
          for (layer <- net.layers) {
            predict_value = layer.forward(predict_value)
            vecs = predict_value :: vecs
          }

          // graph.Histgram2(hist,xs=vecs.reverse,row=2)
          histActor ! Histgram(vecs.reverse, 2)

          predict_value
        } else {
          net.predict(data)
        }
        val t = utils.oneHot(tag.toInt)


        if(tag == argmax(y)) acc += 1

        mixMat(tag.toInt, argmax(y).toInt) += 1

        E += err.calc_cross_entropy_loss(y, t)

        val d = grad.calc_L2_grad(y, t)

        net.update(d)
        ii += 1
      }

      //test
      for((data, tag) <- testset){
        val y = net.predict(data)
        val t = utils.oneHot(tag.toInt)

        if(tag == argmax(y)) tacc += 1

        tmixMat(tag.toInt, argmax(y).toInt) += 1

        tE += err.calc_cross_entropy_loss(y, t)
      }

      E_list  = E  :: E_list
      tE_list = tE :: tE_list

      println(s"$i, E:$E, acc:${acc/datasize*100}%, tE:$tE, tacc:${tacc/testsize*100}%")
      println("mixmat(tag, argmax(y))")
      println("--train--")
      println(mixMat)
      println("--test--")
      println(tmixMat)
      println()

      // graph.Plot(err_rate, Seq(E_list, tE_list).map(l => DenseVector(l.reverse.toArray)),epoch,2)
      // errRateActor ! pltInfo("plot", Seq(E_list, tE_list).map(l => DenseVector(l.reverse.toArray)),Array(s"$epoch","2",""))
      errRateActor ! Line(Seq(E_list, tE_list).map(l => DenseVector(l.reverse.toArray)),epoch,2)
    }

    actorSystem.terminate()
    sys.exit(0)
  }

}
