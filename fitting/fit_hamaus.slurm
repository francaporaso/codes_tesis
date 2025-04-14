#!/bin/bash
#SBATCH --job-name=vvv
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --time=3-00:00:00
#SBATCH --partition=batch
## An email is sent when the job is canceled or finished
## Change "root" for your e-mail address
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=fmcaporaso@mi.unc.edu.ar


. /etc/profile
##source $HOME/.bashrc

##cd /mnt/projects/lensing/HALO_SHAPE/MICEv1.0/codes_HS_micev1.0/

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_py39

srun python fit_void_leastsq.py -sample Rv_6-9 -name smallz_6-9 -ncores 56 -rho hamaus
srun python fit_void_leastsq.py -sample Rv_6-9 -name sz_R_6-9 -ncores 56 -rho hamaus
srun python fit_void_leastsq.py -sample Rv_6-9 -name sz_S_6-9 -ncores 56 -rho hamaus
srun python fit_void_leastsq.py -sample Rv_6-9 -name highz_6-9 -ncores 56 -rho hamaus
srun python fit_void_leastsq.py -sample Rv_6-9 -name hz_R_6-9 -ncores 56 -rho hamaus
srun python fit_void_leastsq.py -sample Rv_6-9 -name hz_S_6-9 -ncores 56 -rho hamaus

srun python fit_void_leastsq.py -sample Rv_9-12 -name smallz_9-12 -ncores 56 -rho hamaus
srun python fit_void_leastsq.py -sample Rv_9-12 -name sz_R_9-12 -ncores 56 -rho hamaus
srun python fit_void_leastsq.py -sample Rv_9-12 -name sz_S_9-12 -ncores 56 -rho hamaus
srun python fit_void_leastsq.py -sample Rv_9-12 -name highz_9-12 -ncores 56 -rho hamaus
srun python fit_void_leastsq.py -sample Rv_9-12 -name hz_R_9-12 -ncores 56 -rho hamaus
srun python fit_void_leastsq.py -sample Rv_9-12 -name hz_S_9-12 -ncores 56 -rho hamaus

srun python fit_void_leastsq.py -sample Rv_12-15 -name smallz_12-15 -ncores 56 -rho hamaus
srun python fit_void_leastsq.py -sample Rv_12-15 -name sz_R_12-15 -ncores 56 -rho hamaus
srun python fit_void_leastsq.py -sample Rv_12-15 -name sz_S_12-15 -ncores 56 -rho hamaus
srun python fit_void_leastsq.py -sample Rv_12-15 -name highz_12-15 -ncores 56 -rho hamaus
srun python fit_void_leastsq.py -sample Rv_12-15 -name hz_R_12-15 -ncores 56 -rho hamaus
srun python fit_void_leastsq.py -sample Rv_12-15 -name hz_S_12-15 -ncores 56 -rho hamaus

srun python fit_void_leastsq.py -sample Rv_15-18 -name smallz_15-18 -ncores 56 -rho hamaus
srun python fit_void_leastsq.py -sample Rv_15-18 -name sz_R_15-18 -ncores 56 -rho hamaus
srun python fit_void_leastsq.py -sample Rv_15-18 -name sz_S_15-18 -ncores 56 -rho hamaus
srun python fit_void_leastsq.py -sample Rv_15-18 -name highz_15-18 -ncores 56 -rho hamaus
srun python fit_void_leastsq.py -sample Rv_15-18 -name hz_R_15-18 -ncores 56 -rho hamaus
srun python fit_void_leastsq.py -sample Rv_15-18 -name hz_S_15-18 -ncores 56 -rho hamaus

srun python fit_void_leastsq.py -sample Rv_18-50 -name smallz_18-50 -ncores 56 -rho hamaus
srun python fit_void_leastsq.py -sample Rv_18-50 -name sz_R_18-50 -ncores 56 -rho hamaus
srun python fit_void_leastsq.py -sample Rv_18-50 -name sz_S_18-50 -ncores 56 -rho hamaus
srun python fit_void_leastsq.py -sample Rv_18-50 -name highz_18-50 -ncores 56 -rho hamaus
srun python fit_void_leastsq.py -sample Rv_18-50 -name hz_R_18-50 -ncores 56 -rho hamaus
srun python fit_void_leastsq.py -sample Rv_18-50 -name hz_S_18-50 -ncores 56 -rho hamaus
