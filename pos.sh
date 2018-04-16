for i in QB RB WR TE; do
   echo $i: $(python baseline.py $i)
done
