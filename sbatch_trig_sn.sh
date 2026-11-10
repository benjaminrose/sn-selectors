#!/bin/bash
#SBATCH --account=sdss
#SBATCH --partition=granite
#SBATCH --qos=granite-freecycle
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --mem=512M
#SBATCH -o %j.log
#SBATCH -e %j.log
#SBATCH --mail-user=ben_rose@baylor.edu
#SBATCH --mail-type=FAIL,BEGIN,END

# `sacct -j *jobid* --format=JobID,Start,End,Elapsed,maxrss`
# gives ram via maxrss
# submit via `sbatch sbatch_trig_sn.sh`

module load uv/sdss
/uufs/chpc.utah.edu/common/home/sdss10/as5work/due/benrose/sn-selectors/trig_sn_young.py
