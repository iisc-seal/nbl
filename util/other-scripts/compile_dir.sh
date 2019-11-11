for dir in $(ls "$1")
do
  ( cd "$1/$dir" && for each in ./*; do gcc -w -std=c99 $each -lm -o "$each.out"; done )
done
