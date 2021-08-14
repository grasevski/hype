#!/bin/sh -eux
while getopts ':a:b:c:' opt; do
	declare $opt=$OPTARG
done

echo c,d,e,score

for d in $(seq 3); do
	for e in $(seq 3); do
		score="$(echo "$a+$b+$c+$d+$e" | bc -l)"
		echo "$c,$d,$e,$score"
	done
done
