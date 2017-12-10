package lstm

import breeze.linalg._
import pll._

object lstm_test {

  var corpus    = "text8.txt"
  var data_size = 1000
  var test_size = 100
  var epoch     = 100
  var node      = 100
  var window    = 10

  /**
    * * Change behavior by "task"
    *    + task:
    *       - "train"   --  Learnig "network" with "data".
    *       - "test"    --  Test "network" with "data".
    *       - "print"   --  Print results predicted "network" by "data".
    *       - otherwise --  Same "test".
    * * Return Value
    *    + ( First, Second )
    *       - First     --  Sum of Error.
    *       - Second    --  Accuracy.
    */
  def one_epoch_at_(task: String)(network: Seq[Layer],
                                  data: List[List[DenseVector[Double]]],
                                  V: word2vec.Vocabrary): (Double, Double) = {

    var num_of_correct = 0
    var num_of_words   = 0
    var E              = 0d

    for (d <- data) {

      var out_vec_list = List[DenseVector[Double]]()
      var count        = 0

      while (count < d.size - 1) {
        var otmp = d(count)
        for (dec <- network) {
          //println(otmp.size)
          otmp = dec.forward(otmp)
        }
        count += 1
        out_vec_list = otmp :: out_vec_list
      }

      val tag = d.tail.map { i =>
        V.indexOf.filter(_._2 == argmax(i)).par.keys.head
      }
      val output = out_vec_list.reverse.map { i =>
        V.indexOf.filter(_._2 == argmax(i)).par.keys.head
      }

      val str_pair = tag zip output
      num_of_words += str_pair.size
      num_of_correct += str_pair.filter { case (t, o) => t == o }.size

      E += (d.tail zip out_vec_list.reverse).map {
        case (t, o) => -sum(t *:* breeze.numerics.log(o))
      }.sum

      task match {
        case "train" =>
          val Ldash = (d.tail zip out_vec_list.reverse).map { case (t, o) => o - t }
          Ldash.reverse.foreach { l =>
            var otmp = l
            for (net <- network.reverse) {
              //println(otmp.size)
              otmp = net.backward(otmp)
            }
          }
          network.map(_.update())
          network.map(_.reset())

        case "print" =>
          println(s"tag:\t${tag.mkString(" ")}\noutput:\t${output.mkString(" ")}\n")

        case "test" | _ =>
      }
    }

    (E, 100d * num_of_correct / num_of_words)
  }

  def main(args: Array[String]) {
    if (args.size == 0) {
      println(s"\nUSAGE:\n\trun lstm.lstm_test [OPTION VALUE]+")
      println(
        s"\nOPTION:\n\t--corpus\n\t--data_size\n\t--test_size\n\t--epoch\n\t--node\n\t--window-size")
      return
    }
    else {
      for (idx <- args.indices) {
        args(idx) match {
          case "--corpus"      => corpus = args(idx + 1)
          case "--data-size"   => data_size = args(idx + 1).toInt
          case "--test-size"   => test_size = args(idx + 1).toInt
          case "--epoch"       => epoch = args(idx + 1).toInt
          case "--node"        => node = args(idx + 1).toInt
          case "--window-size" => window = args(idx + 1).toInt
          case _               =>
        }
      }
      val param_info = s"""
Parameters:
\tcorpus:     \t${corpus}
\tdata_size:  \t${data_size}
\ttest_size:  \t${test_size}
\tepoch:      \t${epoch}
\tnode:       \t${node}
\twindow_size:\t${window}
"""
      println(param_info)
    }

    val V = new word2vec.Vocabrary
    val data = io.Source
      .fromFile(corpus)
      .getLines
      .map(_.trim.split(" ").take(data_size + test_size))
      .toArray
      .flatten
    println("* data loaded")
    V.setVocab(data)
    println("* set vocab")

    val lstm_net = Array(
      new LSTMorg(V.vocab.size, node, "Xavier", 0.1, "Adam", 0.01),
      new Affine(node, V.vocab.size, "Xavier", 0.1, "Adam", 0.01),
      new SoftMax())

    val train_data: List[List[DenseVector[Double]]] =
      (for (idx <- 0 until data_size - 2 * window) yield {
        val l = for (i <- idx until idx + window) yield {
          utils.oneHot(V.indexOf(data(i)), V.vocab.size)
        }
        l.toList
      }).toList

    val test_data: List[List[DenseVector[Double]]] =
      (for (idx <- data_size until data_size + test_size - 2 * window) yield {
        val l = for (i <- idx until idx + window) yield {
          utils.oneHot(V.indexOf(data(i)), V.vocab.size)
        }
        l.toList
      }).toList

    val fig               = breeze.plot.Figure()
    var list_of_train_E   = List[Double]()
    var list_of_test_E    = List[Double]()
    var list_of_train_acc = List[Double]()
    var list_of_test_acc  = List[Double]()

    println("* training start")

    for (i <- 0 until epoch) {

      println(s"  + [${i + 1}/$epoch]")

      val (train_E, train_acc) = one_epoch_at_("train")(lstm_net, train_data, V)
      val (test_E, test_acc)   = one_epoch_at_("test")(lstm_net, test_data, V)

      list_of_train_E = train_E :: list_of_train_E
      list_of_test_E = test_E :: list_of_test_E
      list_of_train_acc = train_acc :: list_of_train_acc
      list_of_test_acc = test_acc :: list_of_test_acc

      println(s"    - train\n\tE: $train_E\n\taccuracy: $train_acc%\n")
      println(s"    - test\n\tE: $test_E\n\taccuracy: $test_acc%\n")
    }

    pll.graph.Line(
      fig,
      Seq(list_of_train_E, list_of_test_E, list_of_train_acc, list_of_test_acc).map(v =>
        DenseVector(v.reverse.toArray)),
      epoch,
      2)(Seq("train_E", "test_E", "train_acc", "test_acc"))

    println("* finish\n")
  }
}
