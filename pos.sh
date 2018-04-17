for i in QB RB WR TE; do
   echo $i: $(python -W ignore baseline.py $i)
done
