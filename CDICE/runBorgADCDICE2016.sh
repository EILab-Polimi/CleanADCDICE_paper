#!/bin/bash

# user input
PROBLEM=ADCDICE2016
NAMEFILE=optADCDICE2016
BORG=/home/acarlino/emodps/borg/borg.exe
# BORG=/Users/angelocarlino/models/emodps/borg/borg.exe
ADAPTIVE=0
ADAPTATION=1
NOBJ=6
EPS=25.0,25.0,20.0,100.0,10.0,20.0
EPS=25.0,10.0,5.0,5.0,5.0,5.0
EPS=25.0,0.05,5.0,10.0,10.0,10.0
# EPS=500.0,0.05,10.0,5.0,10.0,5.0
# EPS=1.0
# Optimization setting
NSEEDS=5
NFE=2500000
if [[ ${ADAPTIVE} == "1" ]]
then
	FOLDER=_DPS
	if [[ ${ADAPTATION} == "1" ]]
	then
		FOLDER+=_AD
		NVAR=60
		NVAR=68
	else
		NVAR=42
	fi
	UB="1"
	LB="0"
	for S in $(seq 2 $NVAR)
	do
		UB="$UB,1"
		LB="$LB,0"
	done
else
	FOLDER=_SO
	if [[ ${ADAPTATION} == "1" ]]
	then
		FOLDER+=_AD
		NVAR=400
	else
		NVAR=200
	fi
	UB="1.2,1"
	LB="0,0"
	for S in $(seq 2 $(($NVAR/2)))
	do
		UB="$UB,1.2,1"
		LB="$LB,0,0"
	done
fi
FOLDER+=_UNC
# NVAR=60 #ADWITCH #60 Agrawala #42 no adaptation 4 nodes
# NVAR=144 #175 #ADWITCH #144 Agrawala 10 nodes
FOLDER+=_${NOBJ}OBJS
# mkdir BorgOutput${FOLDER}
# # optimization
# for SEED in $(seq 1 $NSEEDS)
# do
# OUTFILE=./BorgOutput${FOLDER}/${NAMEFILE}_${SEED}.out
# RUNFILE=./BorgOutput${FOLDER}/rntdynamics_${SEED}.txt
# ${BORG} -n ${NFE} -v ${NVAR} -o ${NOBJ} -s ${SEED} -l ${LB} -u ${UB} -e ${EPS} -f ${OUTFILE} -R ${RUNFILE} -F 100000 ${PROBLEM} 1 &
# done
# echo "optimization terminated"

MYPATH="$(pwd)/BorgOutput${FOLDER}/"
FILENAME=optADCDICE2016
START_COLUMN=$(($NVAR+1))
FINISH_COLUMN=$(($NVAR+$NOBJ))
NUM_SEEDS=$NSEEDS

for ((S=1; S<=$NUM_SEEDS; S++))
do
 echo "Processing SEED $S"
 sed '/^\#/ d' ${MYPATH}${FILENAME}_${S}.out > temp.out
 cat temp.out | cut -d ' ' -f ${START_COLUMN}-${FINISH_COLUMN} > ${MYPATH}${FILENAME}_${S}_obj.txt
 rm temp.out
done

# Extract reference set

JAVA_ARGS="-classpath ./src/moeaframework/MOEAFramework-1.17-Executable.jar:."


java ${JAVA_ARGS} org.moeaframework.util.ReferenceSetMerger -e ${EPS} -o ${MYPATH}${FILENAME}.reference ${MYPATH}${FILENAME}*_obj.txt
