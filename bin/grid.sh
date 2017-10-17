#!/bin/bash

file_path=$1
gw=32
gh=32
echo $file_path
echo $gw
echo $gh

# ext=$(echo file_path##*.)
# extension_list=("png" "jpg")
# 
# flag=false
# 
# for e in extension_list
#    do
#       if [ $e == ext ]; then
#          $flag=true
#       fi
#    done
# 
# if ! $flag; then
#    echo "error!"
#    exit $?
# fi


size=$(identify -format "%wx%h" $file_path)

echo $size

convert -size $size $file_path -fx "i % $gw == 0 || j % $gh == 0 ? 0 : p" grid_$file_path

