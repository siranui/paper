//package pll
import breeze.linalg._
import pll._

object seq2seq {

  val node = 100
  val epoch = 100

//  def oneHot(n: Char) = {
//    var vec = DenseVector.zeros[Double](12)
//    if(n == '+'){
//      vec(10) = 1d
//    } else {
//      vec(n - '0') = 1d
//    }
//    vec
//  }

  def main(args: Array[String]) {
    val ds = 1000
    val ts = 100

    val V = new word2vec.Vocabrary
    val data = io.Source.fromFile("text8.txt").getLines.map(_.trim.split(" ").take(ds+ts)).toArray.flatten
    println("* data loaded")
    V.setVocab(data)
    println("* set vocab")

    // val train_d: List[String] = io.Source.fromFile("src/main/scala/lstm_test/train_d.txt").getLines.take(ds).toList
    // val train_t: List[String] = io.Source.fromFile("src/main/scala/lstm_test/train_t.txt").getLines.take(ds).toList

    // val test_d: List[String] = io.Source.fromFile("src/main/scala/lstm_test/test_d.txt").getLines.take(ts).toList
    // val test_t: List[String] = io.Source.fromFile("src/main/scala/lstm_test/test_t.txt").getLines.take(ts).toList




    val EOS = DenseVector.zeros[Double](V.vocab.size)

    val encoder = Array( new LSTMorg(V.vocab.size,node,"Xavier",0.1,"Adam",0.01))

    var decoder = Array(
          new LSTMorg(V.vocab.size,node,"Xavier",0.1,"Adam",0.01),
          new Affine(node,V.vocab.size,"Xavier",0.1,"Adam",0.01),
          new SoftMax()
          )

    val window = 10

    val tr_d: List[List[DenseVector[Double]]] = (for(idx <- 0 until ds - 2*window) yield {
       var l = (for(i <- idx until idx + window) yield {
          utils.oneHot(V.indexOf(data(i)), V.vocab.size)
       }).toList
       (EOS :: l.reverse).reverse
    }).toList
    val tr_t = (for(idx <- window until ds - window) yield {
       var l = (for(i <- idx until idx + window) yield {
          utils.oneHot(V.indexOf(data(i)), V.vocab.size)
       }).toList
       (EOS :: l.reverse).reverse
    }).toList

    val te_d: List[List[DenseVector[Double]]] = (for(idx <- ds until ds + ts - 2*window) yield {
       var l = (for(i <- idx until idx + window) yield {
          utils.oneHot(V.indexOf(data(i)), V.vocab.size)
       }).toList
       (EOS :: l.reverse).reverse
    }).toList
    val te_t = (for(idx <- ds + window until ds + ts - window) yield {
       var l = (for(i <- idx until idx + window) yield {
          utils.oneHot(V.indexOf(data(i)), V.vocab.size)
       }).toList
       (EOS :: l.reverse).reverse
    }).toList
    // val te_d: List[List[DenseVector[Double]]] = train_d.map(eos :: _.map(oneHot).toList.reverse).reverse
    // val te_t = train_t.map(_.map(oneHot).toList)


    println("* training start")

    for(i <- 0 until epoch){

      println(s"  + [${i+1}/$epoch]")

      var correct = 0
      var correct_test = 0
      var E_train = 0d
      var E_test = 0d

      for((d, t) <- tr_d zip tr_t){
        d.map(encoder(0).forward)

        val (hr, cr) = encoder(0).HRCR()

        var dt = decoder(0).asInstanceOf[LSTMorg]
        dt.Hr = List(hr); dt.Cr = List(cr)
        decoder(0) = dt

        var list = List[DenseVector[Double]]()
        var count = 0
        var loss = List[DenseVector[Double]]()
        do{

          if(list.size == 0){
            list = EOS :: list
          }

          // input for decoder
          // var otmp = list.head
          var otmp = count match{
             case 0 => EOS
             case _ => t(count-1)
          }

          for(dec <- decoder){
            //println(otmp.size)
            otmp = dec.forward(otmp)
          }

          list = otmp :: list
          loss = (otmp - t(count)) :: loss
          // println(s"loss. ${loss.head}")

          //cross-entropy
          E_train += sum(-t(count) * otmp.map(math.log))

          count += 1


        }while(argmax(list.head) != 10 && count < t.size)

        // val str = List("0","1","2","3","4","5","6","7","8","9","+","EOS")

        val tt = t.map{i => V.indexOf.filter(_._2 == argmax(i)).par.keys.head}.mkString(" ")
        // println(s"t: $tt")
        // println(s"loss: ${loss.map(sum(_)).reduce(_+_)}")


        val y = list.reverse.tail.map{i => V.indexOf.filter(_._2 == argmax(i)).par.keys.head}.mkString(" ")
        //println(s"y: $y")

        if(y == tt) correct += 1

        for(l <- 0 until loss.size){
          var tmp = loss(l)
          for(dec <- decoder.reverse){
            tmp = dec.backward(tmp)
          }
        }

        val ddd: LSTMorg = decoder(0).asInstanceOf[LSTMorg]
        val (dn,dc) = ddd.DNDC()
        var et = encoder(0).asInstanceOf[LSTMorg]
        et.dN = dn; et.dC = dc
        encoder(0) = et


        for(_ <- 0 until d.size){
          encoder(0).backward(DenseVector.zeros[Double](node))
        }

        encoder.map(_.update())
        decoder.map(_.update())
        encoder.map(_.reset)
        decoder.map(_.reset)
      }




      //test-start
      for((d,t) <- te_d zip te_t){
        d.map(encoder(0).forward)

        val (hr, cr) = encoder(0).HRCR()

        var dt = decoder(0).asInstanceOf[LSTMorg]
        dt.Hr = List(hr); dt.Cr = List(cr)
        decoder(0) = dt

        var list = List[DenseVector[Double]]()
        var count = 0
        var loss = List[DenseVector[Double]]()
        do{

          if(list.size == 0){
            list = EOS :: list
          }

          var otmp = list.head
          for(dec <- decoder){
            otmp = dec.forward(otmp)
          }

          list = otmp :: list

          //cross-entropy
          E_test += sum(-t(count) * otmp.map(math.log))

          count += 1


        }while(argmax(list.head) != 10 && count < t.size)

        // val str = List("0","1","2","3","4","5","6","7","8","9","+","eos")

        val x = d.map{i => V.indexOf.filter(_._2 == argmax(i)).par.keys.head}.mkString(" ")
        val tt = t.map{i => V.indexOf.filter(_._2 == argmax(i)).par.keys.head}.mkString(" ")

        val y = list.reverse.tail.map{i => V.indexOf.filter(_._2 == argmax(i)).par.keys.head}.mkString(" ")

        println(s"input:\t$x\ntag:\t$tt\noutput:\t$y\n")

        if(y == tt) correct_test += 1


        encoder.map(_.reset)
        decoder.map(_.reset)
      }

      println(s"E_train : ${E_train}")
      println(s"E_test  : ${E_test}")

      println(s"correct rate(train): ${correct.toDouble / tr_d.size}")
      println(s"correct rate(test): ${correct_test.toDouble / te_d.size}")

      // println(s"----- $i -----")

      println(s"$i, ${E_train}, ${E_test}, ${correct.toDouble / tr_d.size * 100}%, ${correct_test.toDouble / te_d.size * 100}%")


    }


  }
}
