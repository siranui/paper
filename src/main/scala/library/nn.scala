package pll

object nn{
  import breeze.linalg._

  def f(){

    val traind = io.Source.fromFile("/home/share/cifar10/train-d.txt").getLines.toArray.take(100)
    val train_d : Array[Array[Double]] = traind.map(_.split(",").map(_.toDouble/256))

    val testd = io.Source.fromFile("/home/share/cifar10/test-d.txt").getLines.toArray.take(100)
    val test_d = testd.map(_.split(",").map(_.toDouble/256))

    val traint = io.Source.fromFile("/home/share/cifar10/train-t.txt").getLines.toList
    val train_t = traint.map(_.split(",").map(_.toInt))

    val testt = io.Source.fromFile("/home/share/cifar10/test-t.txt").getLines.toList
    val test_t = testt.map(_.split(",").map(_.toInt))


    val input : Array[DenseVector[Double]]= train_d.map(a=>(0 until a.size by 3).map(k=>a(k)).toArray).map(b => DenseVector(b))
    val input_test = test_d.map(a=>(0 until a.size by 3).map(k=>a(k)).toArray).map(b => DenseVector(b))

    val layers = List[Layer](/*new Convolution(32*32,30*30,"",0d,"SGD",0.01,3),new sigmoid(),*/new Affine(32*32,10,"uniform",0.01,"SGD",0.01),new SoftMax())


    for(i<- 0 until 10){
    var train_e = 0d
    var traincount = 0
    var trainerror = 0d
      for(j<-0 until input.size){
        var x : DenseVector[Double] = input(j)
        for(layer <- layers){
          x = layer.forward(x)
        }
        val r = DenseVector.fill(10){0d}
        r(train_t(0)(j)) = 1
        var d = x - r
        for(layer <- layers.reverse){
          d = layer.backward(d)
        }
        for(layer <- layers){
          layer.update()
        }
        
        if(train_t(0)(j) == argmax(x)) traincount += 1
      
      train_e -= sum(train_t(0)(j).toDouble *:* max(x,0.1).map(math.log))
      trainerror += train_e
      }
    
      println(traincount)
      println(trainerror)
    }
println("")



    var test_e = 0d
    var testcount = 0
    var testerror = 0d
    for(i<- 0 until input_test.size){
      var x = input_test(i)
      for(layer <- layers){
        x = layer.forward(x)
      }
      for(layer <- layers){
        layer.reset()
      }
        if(test_t(0)(i) == argmax(x)) testcount += 1
       
        test_e -= sum(test_t(0)(i).toDouble *:* max(x,0.0001).map(math.log))
        testerror += test_e
    }
    println(testcount)
    println(testerror)
  }
}
