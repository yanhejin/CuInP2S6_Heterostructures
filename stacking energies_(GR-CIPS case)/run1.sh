#~/tool/miniconda3/envs/virgo/bin/python generate.py , generater the structure before run the lines below
for ii in ??-??/
do
    cd ${ii}
    ln -sf ../INCAR
    ln -sf ../KPOINTS
    ln -sf ../POTCAR
    #ln -sf ../submit_vasp.pbs
    cd ../
done
