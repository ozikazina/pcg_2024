if [[ $# < 2 ]]; then
	echo "Usage: <path to executable> <number of threads> <number of particles>"
	exit -1
fi

ncu -f -o profile\
    --section LaunchStats\
    --section MemoryWorkloadAnalysis\
    --section MemoryWorkloadAnalysis_Chart\
    --section MemoryWorkloadAnalysis_Tables\
    --section Occupancy\
    --section SchedulerStats\
    --section SpeedOfLight\
    --section SpeedOfLight_HierarchicalSingleRooflineChart\
    --section SpeedOfLight_RooflineChart\
    --section WarpStateStats\
    $1 $3 0.01 1 $2 0 4096 128 "generated/$3.dat" out.dat
