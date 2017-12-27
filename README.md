# paper

## 必要条件
* scala
* sbt
* ipython
  + numpy
  + scikit-image
* potrace
* fontforge
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
または、Makefileを同じディレクトリへ配置した上で、
```
make all
```
