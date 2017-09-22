#!/bin/bash

numjobs="$1"

COUNTER=0

while [ $COUNTER -lt $numjobs ]; do
	qsub msu-hpcc-start-btb-job.qsub
	let COUNTER=COUNTER+1
done
