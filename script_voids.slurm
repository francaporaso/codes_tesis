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

##srun ./run_fit_zphot.sh
##srun ./run_fit_zphotw3.sh
##:)srun python forVoid_slice_MICE.py -sample smallz -ncores 10 -nslices 10 -nbins 60 -Rv_min 6. -Rv_max 9. -z_min 0.2 -z_max 0.3 -RIN 0.05 -ROUT 3.
##:)srun python forVoid_slice_MICE.py -sample highz -ncores 10 -nslices 10 -nbins 60 -Rv_min 6. -Rv_max 9. -z_min 0.3 -z_max 0.4 -RIN 0.05 -ROUT 3.
##
##:)srun python forVoid_slice_MICE.py -sample sz_S -ncores 10 -nslices 10 -nbins 60 -Rv_min 6. -Rv_max 9. -z_min 0.2 -z_max 0.3 -RIN 0.05 -ROUT 3. -rho2_max 0.
##:)srun python forVoid_slice_MICE.py -sample hz_S -ncores 10 -nslices 10 -nbins 60 -Rv_min 6. -Rv_max 9. -z_min 0.3 -z_max 0.4 -RIN 0.05 -ROUT 3. -rho2_max 0.
##
##:)srun python forVoid_slice_MICE.py -sample sz_R -ncores 10 -nslices 10 -nbins 60 -Rv_min 6. -Rv_max 9. -z_min 0.2 -z_max 0.3 -RIN 0.05 -ROUT 3. -rho2_min 0.
##:)srun python forVoid_slice_MICE.py -sample hz_R -ncores 10 -nslices 10 -nbins 60 -Rv_min 6. -Rv_max 9. -z_min 0.3 -z_max 0.4 -RIN 0.05 -ROUT 3. -rho2_min 0.
##
##
##:)srun python forVoid_slice_MICE.py -sample smallz -ncores 10 -nslices 10 -nbins 60 -Rv_min 9. -Rv_max 12. -z_min 0.2 -z_max 0.3 -RIN 0.05 -ROUT 3. 
##:)srun python forVoid_slice_MICE.py -sample highz -ncores 10 -nslices 10 -nbins 60 -Rv_min 9. -Rv_max 12. -z_min 0.3 -z_max 0.4 -RIN 0.05 -ROUT 3.
##
##:)srun python forVoid_slice_MICE.py -sample sz_S -ncores 10 -nslices 10 -nbins 60 -Rv_min 9. -Rv_max 12. -z_min 0.2 -z_max 0.3 -RIN 0.05 -ROUT 3. -rho2_min 0.
srun python forVoid_slice_MICE.py -sample hz_S -ncores 56 -nslices 9 -nbins 54 -Rv_min 9. -Rv_max 12. -z_min 0.3 -z_max 0.4 -RIN 0.34 -ROUT 3. -rho2_min 0.
##
##:)srun python forVoid_slice_MICE.py -sample sz_R -ncores 56 -nslices 10 -nbins 60 -Rv_min 9. -Rv_max 12. -z_min 0.2 -z_max 0.3 -RIN 0.05 -ROUT 3. -rho2_max 0.
##srun python forVoid_slice_MICE.py -sample hz_R -ncores 56 -nslices 10 -nbins 60 -Rv_min 9. -Rv_max 12. -z_min 0.3 -z_max 0.4 -RIN 0.05 -ROUT 3. -rho2_max 0.


##:) clemente srun python forVoid_slice_MICE.py -sample smallz -ncores 10 -nslices 10 -nbins 60 -Rv_min 12. -Rv_max 15. -z_min 0.2 -z_max 0.3 -RIN 0.05 -ROUT 3. 
##srun python forVoid_slice_MICE.py -sample highz -ncores 56 -nslices 10 -nbins 60 -Rv_min 12. -Rv_max 15. -z_min 0.3 -z_max 0.4 -RIN 0.05 -ROUT 3.

##:) clementesrun python forVoid_slice_MICE.py -sample sz_S -ncores 56 -nslices 10 -nbins 60 -Rv_min 12. -Rv_max 15. -z_min 0.2 -z_max 0.3 -RIN 0.05 -ROUT 3. -rho2_min 0.
##srun python forVoid_slice_MICE.py -sample hz_S -ncores 56 -nslices 10 -nbins 60 -Rv_min 12. -Rv_max 15. -z_min 0.3 -z_max 0.4 -RIN 0.05 -ROUT 3. -rho2_min 0.

##:) sersicsrun python forVoid_slice_MICE.py -sample sz_R -ncores 56 -nslices 10 -nbins 60 -Rv_min 12. -Rv_max 15. -z_min 0.2 -z_max 0.3 -RIN 0.05 -ROUT 3. -rho2_max 0.
##srun python forVoid_slice_MICE.py -sample hz_R -ncores 56 -nslices 10 -nbins 60 -Rv_min 12. -Rv_max 15. -z_min 0.3 -z_max 0.4 -RIN 0.05 -ROUT 3. -rho2_max 0.


##:)srun python forVoid_slice_MICE.py -sample smallz -ncores 10 -nslices 10 -nbins 60 -Rv_min 15. -Rv_max 18. -z_min 0.2 -z_max 0.3 -RIN 2. -ROUT 3. 
##srun python forVoid_slice_MICE.py -sample highz -ncores 56 -nslices 10 -nbins 60 -Rv_min 15. -Rv_max 18. -z_min 0.3 -z_max 0.4 -RIN 0.05 -ROUT 3.

##:)srun python forVoid_slice_MICE.py -sample sz_S -ncores 56 -nslices 10 -nbins 60 -Rv_min 15. -Rv_max 18. -z_min 0.2 -z_max 0.3 -RIN 0.05 -ROUT 3. -rho2_min 0.
##srun python forVoid_slice_MICE.py -sample hz_S -ncores 56 -nslices 10 -nbins 60 -Rv_min 15. -Rv_max 18. -z_min 0.3 -z_max 0.4 -RIN 0.05 -ROUT 3. -rho2_min 0.

##:) srun python forVoid_slice_MICE.py -sample sz_R -ncores 56 -nslices 10 -nbins 60 -Rv_min 15. -Rv_max 18. -z_min 0.2 -z_max 0.3 -RIN 0.05 -ROUT 3. -rho2_max 0.
##srun python forVoid_slice_MICE.py -sample hz_R -ncores 56 -nslices 10 -nbins 60 -Rv_min 15. -Rv_max 18. -z_min 0.3 -z_max 0.4 -RIN 0.05 -ROUT 3. -rho2_max 0.


##:) srun python forVoid_slice_MICE.py -sample smallz -ncores 56 -nslices 10 -nbins 60 -Rv_min 18. -z_min 0.2 -z_max 0.3 -RIN 0.05 -ROUT 3. 
##  srun python forVoid_slice_MICE.py -sample highz -ncores 56 -nslices 10 -nbins 60 -Rv_min 18. -z_min 0.3 -z_max 0.4 -RIN 0.05 -ROUT 3.
##  
##:) srun python forVoid_slice_MICE.py -sample sz_S -ncores 56 -nslices 10 -nbins 60 -Rv_min 18. -z_min 0.2 -z_max 0.3 -RIN 0.05 -ROUT 3. -rho2_min 0.
##  srun python forVoid_slice_MICE.py -sample hz_S -ncores 56 -nslices 10 -nbins 60 -Rv_min 18. -z_min 0.3 -z_max 0.4 -RIN 0.05 -ROUT 3. -rho2_min 0.
##  
##:) srun python forVoid_slice_MICE.py -sample sz_R -ncores 56 -nslices 10 -nbins 60 -Rv_min 18. -z_min 0.2 -z_max 0.3 -RIN 0.05 -ROUT 3. -rho2_max 0.
##  srun python forVoid_slice_MICE.py -sample hz_R -ncores 56 -nslices 10 -nbins 60 -Rv_min 18. -z_min 0.3 -z_max 0.4 -RIN 0.05 -ROUT 3. -rho2_max 0.

