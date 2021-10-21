import h5py
import numpy as np
import os

def get_all_results():
    for dirpath, _, files in os.walk("results"):
        for f in files:
            if f.endswith('hdf5'):
                yield h5py.File(os.path.join(dirpath, f), 'r')

if __name__ == "__main__":
    for f in get_all_results():
        g = h5py.File(os.path.join("data", f"{f.attrs['dataset']}.hdf5"), 'r')
        try:
            abs_err = np.mean(np.abs(np.array(f['errors']) - np.array(g['kde.01']))/np.array(g['kde.01']))
            query_time = np.mean(f['times'])
            print(f'{f.attrs["algorithm"]}\t{abs_err}\t{query_time}')
        except:
            print(f"Couldn't process {f}")


