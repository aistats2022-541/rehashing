from itertools import product
import yaml
import os
import sys
import subprocess
import re
import h5py
import numpy

def get_result_fn(dataset, algorithm, eps, mu):
    dir_name = os.path.join("results", dataset, algorithm, mu)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return os.path.join(dir_name, f"{eps}.hdf5")

def invoke(dataset, algorithm, eps):
    if algorithm == "hbe":
        return f'{sys.argv[2]} resources/data/{dataset}.conf gaussian {eps}'
    if algorithm == 'rs':
        return f'{sys.argv[2]} resources/data/{dataset}.conf gaussian {eps} true'
    print(f'Cannot invoke {algorithm}')

def write_result(result, ds, algorithm, eps, mu):
    fn = get_result_fn(ds, algorithm, eps, mu)
    regex = r"RESULT id=(\d+) err=(\d+.\d+) samples=(\d+) time=(\d+.\d+)"
    ids = []
    errs = []
    samples = []
    times = []
    for line in result.split("\n"):
        if m := re.match(regex, line):
            ids.append(int(m.group(1)))
            errs.append(float(m.group(2)))
            samples.append(int(m.group(3)))
            times.append(float(m.group(4)))
    f = h5py.File(fn, 'w')
    f.attrs['dataset'] = ds
    f.attrs['algorithm'] = algorithm
    f.attrs['params'] = f'eps={eps}'
    f.attrs['mu'] = mu
    f.create_dataset('ids', data=numpy.array(ids))
    f.create_dataset('errors', data=numpy.array(errs))
    f.create_dataset('samples', data=numpy.array(samples))
    f.create_dataset('times', data=numpy.array(times))
    f.close()


if __name__ == "__main__":

    mu = "0.01"
    if len(sys.argv) != 3:
        print(f'Usage: python3 {sys.argv[0]} <exp-file> <hbe-binary>')
        exit(1)

    with open(sys.argv[1], 'r') as f:
        expfile = yaml.load(f, Loader=yaml.Loader)

    exps = [(ds, eps, algorithm)
            for ds, eps, algorithm in product(expfile['dataset'], expfile['eps'], expfile['algorithms'])
            if not os.path.exists(get_result_fn(ds, algorithm, eps, mu))]
    for ds, eps, algorithm in exps:
        print(f'Running {algorithm} on {ds} with eps={eps}')
        print(invoke(ds, algorithm, eps))
        result = subprocess.run(invoke(ds, algorithm, eps).split(), stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        write_result(result.stdout.decode('utf-8').strip(), ds, algorithm, eps, mu)









