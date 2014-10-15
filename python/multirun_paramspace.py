import os
import subprocess
import numpy as np
import pickle

# Fine grid
#Qsearch = np.arange(0.745, 0.760, 0.001)
#Rmsearch = np.arange(4.84, 4.90, 0.01)

#Finest grid
#Qsearch = np.arange(0.749, 0.751, 0.0005)
#Rmsearch = np.arange(4.874, 4.879, 0.0005)

# Coarsest grid (for comparison with Umurhan+_)
#Qsearch = np.arange(0.1, 1.5, 0.1)
#Rmsearch = np.arange(4.6, 5.4, 0.1)

# coarse grid for Rm ~ 50
#Qsearch = np.arange(0.1, 1.5, 0.1)

#Qsearch = np.arange(1, 15, 1.0)
#Rmsearch = np.arange(46, 54, 1.0)

#Qsearch = np.arange(1.5, 3.5, 0.25)
#Rmsearch = np.arange(43, 46, 0.2)

#Qsearch = np.arange(2.0, 2.5, 0.1)
#Rmsearch = np.arange(45, 45.5, 0.1)#Q=2.3, Rm=45.1

# coarse grid for Rm ~ 500
#Qsearch = np.arange(5, 40, 5) #Q 0-150 Rm 360-460, #Q 10-140 Rm 460-530
#Rmsearch = np.arange(200, 600, 10)

#Pm = 0.0001, so Rm ~ 0.5
Rmsearch = np.arange(0.2, 1.2, 0.1)
Qsearch = np.arange(0.45, 0.55, 0.01) 

run_script = "multirun_linear_MRI.py"

processes = {}
for Rm in Rmsearch:
    for Q in Qsearch: 
    
        print("Starting process")
        proc_args = ["python3",os.path.join(os.getcwd(),run_script), str(Rm), str(Q)]
        processes[Rm, Q] = subprocess.Popen(proc_args, stdout=subprocess.PIPE)
        processes[Rm, Q].wait()

print(processes)
results = {}
for k,proc in processes.items():
    res = proc.communicate()
    #print(res)
    if res is not None:
        str = res[0].decode("utf-8")
        print("str", str)
        results[k] = complex(str)

#print(results)

pickle.dump(results, open("multirun_Rm_50.p", "wb"))

#np.save("results_Qrange"+str(Qsearch[0])+"_to_"+str(Qsearch[-1])+"Rrange"+str(Rmsearch[0])+"_to_"+str(Rmsearch[-1])+".npy", results)
#np.save("results_test.npy", results)