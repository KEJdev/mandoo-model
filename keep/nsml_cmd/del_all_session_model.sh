#!/bin/bash
for (( c=140; c<=197; c++ ))
do
    nsml model rm team_17/ir_ph2/$c "*"
done
