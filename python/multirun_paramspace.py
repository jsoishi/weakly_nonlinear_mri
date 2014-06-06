import os
import subprocess
import numpy as np
import pickle

#Qsearch = np.arange(0.710, 0.760, 0.002)
#Rmsearch = np.arange(4.88, 4.95, 0.01)

Qsearch = np.arange(0.1, 1.5, 0.1)
Rmsearch = np.arange(4.6, 5.4, 0.1)

#Qsearch = [0.746, 0.748, 0.750, 0.752, 0.754]
#Rmsearch = [0.48, 0.49, 0.50]

run_script = "multirun_linear_MRI.py"

processes = {}
for Rm in Rmsearch:
    for Q in Qsearch: 
    
        print("Starting process")
        proc_args = ["python3",os.path.join(os.getcwd(),run_script), str(Rm), str(Q)]
        processes[Rm, Q] = subprocess.Popen(proc_args, stdout=subprocess.PIPE)
        processes[Rm, Q].wait()
        print("Ending process")

        

results = {}
for k,proc in processes.items():
    res = proc.communicate()
    if res is not None:
        str = res[0].decode("utf-8")
        results[k] = complex(str)

print(results)

pickle.dump(results, open("results_coarse_Co_fixed.p", "wb"))

#np.save("results_Qrange"+str(Qsearch[0])+"_to_"+str(Qsearch[-1])+"Rrange"+str(Rmsearch[0])+"_to_"+str(Rmsearch[-1])+".npy", results)
#np.save("results_test.npy", results)