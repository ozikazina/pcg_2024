if [[ $# < 2 ]]; then
	echo "Usage: <path to executable> <number of threads> <number of particles>"
	exit -1
fi

nsys profile -f --trace=cuda --output=nsysreport $1 $3 0.01 1 $2 0 4096 128 "generated/$3.dat" out.dat