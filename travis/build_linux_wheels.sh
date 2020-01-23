#!/bin/bash

mkdir -p /io/wheelhouse/manylinux 

pip_paths=(
	"/opt/python/cp36-cp36m/bin/pip"
	"/opt/python/cp37-cp37m/bin/pip"
	)

for pip_path in "${pip_paths[@]}"; do
    	echo "$pip_path"
	$pip_path wheel /io -w /io/wheelhouse/
done



for whl in io/wheelhouse/cellfinder*.whl; do
    auditwheel repair "$whl" -w /io/wheelhouse/manylinux
done
