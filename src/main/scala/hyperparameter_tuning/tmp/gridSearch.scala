package hyperparameter_tune

import breeze.linalg._
// import pll._
import pll.typeAlias._
import akka.actor.{Actor, /* ActorLogging, */ Props}
import akka.actor.ActorSystem
import akka.pattern.ask
import akka.util.Timeout
import scala.concurrent.duration._

// 1. 学習データ・テストデータの読み込み
// 2. ネットワークの作成
// 3. 探索するハイパーパラメータを列挙
// 4. 3から、ハイパーパラメータを選択
// 5. 選択したパラメータで学習
// 6. 4-5を繰り返す
// 7. 正解率等で評価し、評価の高くなったハイパーパラメータを出力

// 3,4,7の責任を持つ
object gridSearch {

  def main(args: Array[String]) {

    mock.args_process(args)

    // 3. 探索するハイパーパラメータを列挙
    val distrs         = Seq("Xavier", "He", "Gaussian", "Uniform")
    val SDs            = linspace(0.01, 10, 4).toArray.toSeq
    val update_methods = Seq("SGD", "AdaGrad", "RMSProp", "Adam")
    val lrs            = linspace(0.01, 10, 4).toArray.toSeq

    println("--- data loading ---")
    val Seq((train_data, train_tag), (test_data, test_tag)) = mock.data_load()

    val train_set: DATASET = train_data zip train_tag
    val test_set: DATASET  = test_data zip test_tag

    var tAccs = Map[(String, Double, String, Double), Double]()

    // for {
    //   distr <- distrs
    //   SD <- SDs
    //   update_method <- update_methods
    //   lr <- lrs
    //   if(!(distr == "Xavier" && SD != SDs.head))
    //   if(!(distr == "He" && SD != SDs.head))
    // } yield {
    //   // println("--- network initialize ---")
    //   val net = cifar10_net(distr, SD, update_method, lr)

    //   // println("--- training start ---")
    //   val (train_err, train_acc, test_err, test_acc) = mock.train(net, train_set, test_set)

    //   println(distr, SD, update_method, lr)
    //   println(s"$train_err\t${train_acc}\t$test_err\t${test_acc}")
    //   tAccs += (distr, SD, update_method, lr) -> test_acc
    // }

    val system           = ActorSystem("system")
    val grid_seacher     = system.actorOf(Props[searcher], "grid-searcher")
    implicit val timeout = Timeout(10000.milliseconds)

    for {
      distr         <- distrs
      SD            <- SDs
      update_method <- update_methods
      lr            <- lrs
      if (!(distr == "Xavier" && SD != SDs.head))
      if (!(distr == "He" && SD != SDs.head))
    } yield {
      val test_acc = grid_seacher ? searcher.Conditions(
        distr,
        SD,
        update_method,
        lr,
        train_set,
        test_set)
    }
    println()
    println(
      "----------------------------------------------------------------------------------------------------")
    tAccs.toArray.sortBy(_._2).reverse.take(10).toList.foreach(println)
    // system.stop(grid_seacher)
    // system.terminate
  }
}

object searcher {
  case class Conditions(distr: String,
                        standerd_deviation: Double,
                        update_method: String,
                        lr: Double,
                        train_set: DATASET,
                        test_set: DATASET)
  case class Response(result: Double)
}

class searcher extends Actor {
  import searcher._

  def receive = {
    case Conditions(distr, standerd_deviation, update_method, lr, train_set, test_set) =>
      // println("--- network initialize ---")
      val net = cifar10_net(distr, standerd_deviation, update_method, lr)

      // println("--- training start ---")
      val (train_err, train_acc, test_err, test_acc) = mock.train(net, train_set, test_set)

      println(s"$distr, $standerd_deviation, $update_method, $lr")
      println(s"$train_err\t${train_acc}\t$test_err\t${test_acc}")

      sender() ! Response(test_acc)
  }
}
