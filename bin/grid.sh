#!/bin/bash

file_path=$1
size=$(identify -format "%wx%h" $file_path)
gw=32 # grid width
gh=32 # grid height

echo "target:      $file_path"
echo "image size:  $size"
echo "grid width:  $gw"
echo "grid height: $gh"

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


convert -size $size $file_path -fx "i % $gw == 0 || j % $gh == 0 ? 0 : p" grid_$file_path

