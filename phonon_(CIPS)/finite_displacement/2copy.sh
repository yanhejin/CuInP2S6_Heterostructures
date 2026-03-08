for j in {001..060}
do
   cd $j
   cp ../INCAR ./
   cp ../KPOINTS ./
   cp ../POTCAR ./
   cd ../
done
