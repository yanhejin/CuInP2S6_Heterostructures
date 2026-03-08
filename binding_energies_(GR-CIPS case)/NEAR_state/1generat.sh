for j in *
do
if [ -d "$j" ]; then 
 cd $j
 echo $j
 cd total
 ln -s ../../INCAR ./INCAR
 ln -s ../../KPOINTS ./KPOINTS
 cp ../../potcar.sh ./potcar.sh
 sh potcar.sh  $(sed -n 6p ./POSCAR)
 echo "total done"
 cd ../
 cd lower
 ln -s ../../INCAR ./INCAR
 ln -s ../../KPOINTS ./KPOINTS
 cp ../../potcar.sh ./potcar.sh
 sh potcar.sh  $(sed -n 6p ./POSCAR)
 echo "lower done"
 cd ../
 cd upper
 ln -s ../../INCAR ./INCAR
 ln -s ../../KPOINTS ./KPOINTS
 cp ../../potcar.sh ./potcar.sh
 sh potcar.sh  $(sed -n 6p ./POSCAR)
 echo "upper done"
 cd ../../
fi
done
