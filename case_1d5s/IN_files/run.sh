#PBS -l nodes=1:ppn=12
#####PBS -l nodes=node17:ppn=12+node18:ppn=12+node19:ppn=12+node20:ppn=12
#####PBS -l nodes=node07:ppn=12
#PBS -e stderr.err
#PBS -o stdout.out
#PBS -N Bp_slope2

NPROCS=`wc -l < $PBS_NODEFILE`

cd $PBS_O_WORKDIR

echo "Starting Run"
echo "using $NPROCS CPUs"
echo "working dir is:" `pwd`
echo `date`

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/netcdf4-intel/lib:/opt/hdf5-intel/lib

time mpirun -hostfile $PBS_NODEFILE -n $NPROCS ./romsM ./roms_fhl1_new.in > temp.out


###time mpiexec ./oceanM ./ocean_tidalflat.in > temp.out

echo "Finish Run"
echo `date`
#-----------------------------------------------
