# paper

## 必要条件
* scala
  + breeze (version 0.13)
  + breeze-natives (version 0.13)
  + breeze-viz (version 0.13)
* sbt
* python3
  + numpy
  + scikit-image
* [potrace](http://potrace.sourceforge.net/)
* [fontforge](https://fontforge.github.io/en-US/)
* make

## 出力データを画像に変換

正方形なら
```sh
ipython font.py {data_file} {width}
```
長方形なら
```sh
ipython font.py {data_file} {width} {height}
```

## 画像にグリッド線を引く
```
bin/grid.sh {image_file}
```

## 画像(png)からフォント(ttf)を生成する
u0041.pngのように、ファイル名をその文字のasciiコードで表した画像が有るディレクトリで
```
make -f bin/Makefile all
```
または、Makefileを同じディレクトリへ配置した上で
```
make all
```
[asciiコード](https://ja.wikipedia.org/wiki/ASCII)
