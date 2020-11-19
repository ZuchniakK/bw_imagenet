#!/bin/bash -l
## Nazwa zlecenia
#SBATCH -J resnetbw50
## Liczba alokowanych węzłów
#SBATCH -N 1
## Liczba zadań per węzeł (domyślnie jest to liczba alokowanych rdzeni na węźle)
#SBATCH --ntasks-per-node=1
## Ilość pamięci przypadającej na jeden rdzeń obliczeniowy (domyślnie 5GB na rdzeń)
#SBATCH --mem-per-cpu=20GB
## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=12:00:00
## Nazwa grantu do rozliczenia zużycia zasobów
#SBATCH -A plgdaisnet
## Specyfikacja partycji
#SBATCH --partition=plgrid-gpu-v100
#SBATCH --gres=gpu:2
## Plik ze standardowym wyjściem
#SBATCH --output="resnetbw50.out"
## Plik ze standardowym wyjściem błędów
#SBATCH --error="resnetbw50.err"


## przejscie do katalogu z ktorego wywolany zostal sbatch
cd $SLURM_SUBMIT_DIR

srun /bin/hostname

module plgrid/libs/python-numpy/1.18.4-python-3.8
module load plgrid/libs/tensorflow-gpu/2.3.1-python-3.8
export LD_LIBRARY_PATH=/net/people/plgurvanish/cuda/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/net/scratch/people/plgurvanish/bw_imagenet:$PYTHONPATH

python3 train.py ResNet50 128