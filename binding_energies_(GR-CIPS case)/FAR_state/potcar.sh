#!/usr/bin/env bash
# Create a GGA_PAW POTCAR file
# by BigBro  
# To Use it： potcar.sh Cu C H O

# Define local potpaw_GGA pseudopotentialrepository:
repo="/data/home/yc17806/back/potential/vasp_potential/potpaw_PBE"

# Check if older version of POTCAR ispresent
if [ -f POTCAR ] ; then
 mv -f POTCAR old-POTCAR
 echo " ** Warning: old POTCAR file found and renamed to 'old-POTCAR'."
fi

# Main loop - concatenate the appropriatePOTCARs (or archives)
for i in $*
do
 if test -f $repo/$i/POTCAR ; then
  cat $repo/$i/POTCAR>> POTCAR
 else
 echo " ** Warning: No suitable POTCAR for element '$i' found!! Skipped thiselement."
 fi
done
