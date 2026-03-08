for ii in ??-??/
do
    cd ${ii}
    echo -e $ii "\t" $(tail -n 1 ./OSZICAR | awk '{print $5}') 
    cd ../
done
