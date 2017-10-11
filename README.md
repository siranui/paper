# paper


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
convert 
  \ -size {width}x{height} {file_name}
  \ -fx "i % {grid_width} == 0 || j % {grid_height} == 0 ? 0 : p"
  \ {converted_file_name} 
```
