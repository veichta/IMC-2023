sync:
	rsync -auv --progress . ${USR}@euler.ethz.ch:/cluster/home/${USR}/code/IMC-2023 --exclude=image-matching-challenge-2023 --exclude=*.h5 --exclude=*.csv --exclude outputs
	rsync -auv --progress ${USR}@euler.ethz.ch:/cluster/scratch/${USR}/outputs .

wheel_hloc:
	python ext_deps/Hierarchical-Localization/setup.py bdist_wheel
	mv dist/*.whl wheels/
	rm -rf dist build


wheel_dioad:
	python ext_deps/dioad/setup.py bdist_wheel
	mv dist/*.whl wheels/
	rm -rf dist build