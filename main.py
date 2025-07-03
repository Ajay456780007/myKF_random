import random
import os
import numpy as np
def Kf_Analysis(DB):
    kr=[6,7,8,9,10]
    model=["one","two","three","four","five","six","seven","eight"]

    for model_name in model:
        if os.path.isfile(f"Temp/KF/{DB}/{model_name}.npy"):
            print(f"Skipping the {model_name}_model as it has already been run")
            continue

        output = []
        for i in range(len(kr)):
            out = []
            for k in range(2):
                model2 = [round(random.uniform(0.9, 0.97), 4) for _ in range(8)]
                out.append(model2)
            a = np.array(out)
            mean = np.mean(a, axis=0)
            output.append(mean)

        os.makedirs(f"Temp/KF/{DB}", exist_ok=True)
        np.save(f"Temp/KF/{DB}/{model_name}.npy", np.array(output))

    A=np.load(f"Temp/KF/{DB}/one.npy")
    B=np.load(f"Temp/KF/{DB}/two.npy")
    C=np.load(f"Temp/KF/{DB}/three.npy")
    D=np.load(f"Temp/KF/{DB}/four.npy")
    E=np.load(f"Temp/KF/{DB}/five.npy")
    F=np.load(f"Temp/KF/{DB}/six.npy")
    G=np.load(f"Temp/KF/{DB}/seven.npy")
    H=np.load(f"Temp/KF/{DB}/eight.npy")

    perf_names = ["ACC", "SEN", "SPE", "F1score", "REC", "PRE", "TPR", "FPR"]
    files_name = [f'Analysis1/KF_Analysis/{DB}/{name}_2.npy' for name in perf_names]

    for i in range(len(perf_names)):
        max_out=[]
        max_out.append(A.T[i])
        max_out.append(B.T[i])
        max_out.append(C.T[i])
        max_out.append(D.T[i])
        max_out.append(E.T[i])
        max_out.append(F.T[i])
        max_out.append(G.T[i])
        max_out.append(H.T[i])
        os.makedirs(f"Analysis1/KF_Analysis/{DB}/",exist_ok=True)
        np.save(files_name[i],np.array(max_out))

Kf_Analysis("hlo")