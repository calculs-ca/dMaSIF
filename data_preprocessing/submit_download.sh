#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH -D /home/jflucier/projects/def-rodrigu1/tmp/program/dMaSIF/data_preprocessing
#SBATCH -o /home/jflucier/projects/def-rodrigu1/tmp/program/dMaSIF/data_preprocessing/log/dnl_-%A_%a.out
#SBATCH --time=24:00:0
#SBATCH --mem=30G
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A def-xroucou
#SBATCH --mail-user=$JOB_MAIL
#SBATCH -J dnl

export export PATH=/home/jflucier/projects/def-rodrigu1/tmp/program/MolProbity/bin/linux:$PATH
export REDUCE_HET_DICT=/home/jflucier/projects/def-rodrigu1/tmp/program/MolProbity/lib/reduce_wwPDB_het_dict.txt
export MASIF_HOME=/home/jflucier/projects/def-rodrigu1/tmp/program/dMaSIF
cd $MASIF_HOME
source env3.7/bin/activate

export __INFILE=$(ls /home/jflucier/projects/def-rodrigu1/tmp/program/dMaSIF/data_preprocessing/split/pdbid_part.* | awk "NR==$SLURM_ARRAY_TASK_ID")

echo "running download script for ${__INFILE}"
python my.download_pdb.py --pdb_list ${__INFILE}

echo 'done!'

