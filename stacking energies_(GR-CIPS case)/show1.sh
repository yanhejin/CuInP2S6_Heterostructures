ls ??-??/OUTCAR | xargs -I {} bash -c '(grep "accounting informations" {} >& /dev/null || echo {})'
