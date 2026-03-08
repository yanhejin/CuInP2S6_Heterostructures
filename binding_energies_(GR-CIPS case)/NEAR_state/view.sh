for i in *
do
if [ -d "$i" ]
  then python layerdistance_grephene_total.py -dics=$i
fi
done
