#!/bin/bash

run_map() {
    local map=$1
    local param1=$2
    local param2=$3
    local param3=$4
    local output=""
    for i in {1..5}
    do
        result=$(./main_gpu.exe $map $param1 $param2 $param3)
        output+="$result, "
    done

    # Remove the trailing comma and space
    output=$(echo $output | sed 's/, $//')
    echo "$map, $param1, $param2, $param3, $output"
}

maps=(32 64 128 192 256 320 384 448 512 640 768 896 1024)

for map in "${maps[@]}"
do
    run_map $map 8 8 8
done