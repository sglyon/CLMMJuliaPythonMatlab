#!/bin/bash
#
#SBATCH --job-name=NK_JMP_matlab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=6:00:00
#SBATCH --mail-user=sgl290@nyu.edu
#SBATCH --output=NK_JMP_matlab_start_deg2_loop.out

module purge
module load matlab/2017b

echo
echo "Hostname: $(hostname)"
echo "Job starts: $(date)"
echo

cat<<EOF | matlab -nodisplay
disp('Hello world from matlab!');
try
	disp('About to call for_paper');
    for_paper();
disp('I think I successfully called for_paper');
	
catch err
    fprintf('\n\nTime: %s\n', datestr(datetime('now')));
    fprintf('Matlab error: %s\n', err.message);
    exit(1);
end
EOF

matlab_status=$?
echo
echo
echo "Job ends: $(date)"
exit $matlab_status
