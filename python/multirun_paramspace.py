import os
import subprocess
import numpy as np
import pickle

# Fine grid
#Qsearch = np.arange(0.745, 0.760, 0.001)
#Rmsearch = np.arange(4.84, 4.90, 0.01)

#Finest grid
Qsearch = np.arange(0.749, 0.751, 0.0005)
Rmsearch = np.arange(4.874, 4.879, 0.0005)

# Coarsest grid (for comparison with Umurhan+_)
#Qsearch = np.arange(0.1, 1.5, 0.1)
#Rmsearch = np.arange(4.6, 5.4, 0.1)


run_script = "multirun_linear_MRI.py"

processes = {}
for Rm in Rmsearch:
    for Q in Qsearch: 
    
        print("Starting process")
        proc_args = ["python3",os.path.join(os.getcwd(),run_script), str(Rm), str(Q)]
        processes[Rm, Q] = subprocess.Popen(proc_args, stdout=subprocess.PIPE)
        processes[Rm, Q].wait()

results = {}
for k,proc in processes.items():
    res = proc.communicate()
    if res is not None:
        str = res[0].decode("utf-8")
        results[k] = complex(str)

print(results)

pickle.dump(results, open("results_finegrid6.p", "wb"))

#np.save("results_Qrange"+str(Qsearch[0])+"_to_"+str(Qsearch[-1])+"Rrange"+str(Rmsearch[0])+"_to_"+str(Rmsearch[-1])+".npy", results)
#np.save("results_test.npy", results)