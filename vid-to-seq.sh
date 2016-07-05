mkdir frames
ffmpeg -i test1.mp4 -f image2 frames/frame-%3d.png
mkdir tiles
for file in frames/*.png; do convert -crop 64x64 +repage $file tiles/`basename $file .png`-tile%02d.png; done
tar -cvf dataz.tar tiles/
java -jar tar-to-seq/tar-to-seq.jar data.tar carvid.seq