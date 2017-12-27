# Image and label files are created in the called directory.
#
# ./
# └ ${font_name}/
#   ├ image/
#   │ ├ ${font_name}_${char}_${asciicode}.png
#   │ ├ ${font_name}_${char}_${asciicode}.png
#   │ └ ${font_name}_${char}_${asciicode}.png
#   ├ ${font_name}-d.txt                        # data written by 0-256
#   └ ${font_name}-t.txt                        # label
#
# USAGE: ipython font_to_png_and_txt.py FONT_FILE_PATH [WIDTH] [HEIGHT]

import subprocess as sp
import sys
from skimage import io

args = sys.argv

PATH = args[1]

# decide the image size
if len(args) == 4:
    WIDTH = args[2]
    HEIGHT = args[3]
elif len(args) == 3:
    WIDTH = HEIGHT = args[2]
else:
    WIDTH = HEIGHT = '64'


font_name = PATH.split("/")[-1].split(".")[0]

# characters
# atoz = [chr(i) for i in range(97,97+26)]
# AtoZ = [chr(i) for i in range(65,65+26)]
# num = [chr(i) for i in range(ord('0'),ord('9')+1)]
Ascii = [chr(i) for i in range(32, 127)] 

# list = atoz + AtoZ + num
list = Ascii

# create save directory
sp.run(["mkdir", "-p", font_name+"/image"])

# data
f = open(font_name + '/' + font_name + '-d.txt', 'w')
# label
g = open(font_name + '/' + font_name + '-t.txt', 'w')

# create image file
for num, c in enumerate(list):
    if c is '/':
      file_name = font_name+str(ord(c))+"_slash_.png"
    elif c is '\\':
      file_name = font_name+str(ord(c))+"_back_slash_.png"
    else:
      file_name = font_name+str(ord(c))+"_'"+c+"'_"+".png"


    if c is not '\\':
      sp.run(["convert", "-font", PATH, "-size", WIDTH+"x"+HEIGHT, "-gravity", "Center", "label:"+c, font_name+"/image/"+file_name ])
    else:
      sp.run(["convert", "-font", PATH, "-size", WIDTH+"x"+HEIGHT, "-gravity", "Center", "label:"+c+c, font_name+"/image/"+file_name ])


    img = io.imread(font_name + "/image/" + file_name)
    f.write( ','.join( map(str, img.reshape(-1)) ) )
    f.write('\n')
    g.write(c+'\n')

    sys.stdout.write("\r{}: [*{}{}]".format(font_name, "*"*num, "-"*(len(list)-(num+1))))
    sys.stdout.flush()

f.close()
g.close()

print()
