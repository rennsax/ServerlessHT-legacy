import itertools
import subprocess

import numpy as np

if __name__ == "__main__":
    design_space = {
        "batch_size": np.array([16, 32, 64, 128]),
        "learning_rate": np.linspace(0.001, 0.1, 16),
        "momentum": np.linspace(0.6, 0.9, 4),
    }

    hyperparameter_list = [*itertools.product(*design_space.values())]

    process_list = []
    for idx, (bs, lr, m) in enumerate(hyperparameter_list):
        command = ["python3", "lenet.py"]
        command.append("-g")
        command.append(str(idx % 2))
        command.append("-b")
        command.append(str(bs))
        command.append("-l")
        command.append(str(lr))
        command.append("-m")
        command.append(str(m))
        command.append("--output")
        command.append("output/lenet-{}".format(idx % 4))
        fd = open("output/lenet-{}.stdout".format(idx % 4), "w")
        process_list.append((subprocess.Popen(command, stdout=fd, stderr=fd), fd))

        if idx % 4 == 3:
            for p, fd in process_list:
                p.wait()
                fd.close()
            process_list.clear()
