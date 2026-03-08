for j in {001..060}
do
   mkdir $j
   mv ./POSCAR-"$j" $j/POSCAR
done
