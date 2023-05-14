sync:
	rsync -auv --progress . ${USR}@euler.ethz.ch:/cluster/home/${USR}/code/IMC-2023 --exclude=image-matching-challenge-2023 --exclude=*.h5 --exclude=*.csv --exclude outputs
	rsync -auv --progress ${USR}@euler.ethz.ch:/cluster/home/${USR}/code/IMC-2023/outputs .