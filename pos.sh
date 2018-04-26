for i in QB RB WR TE; do
   echo $i: $(python -W ignore $1 $i)
done
