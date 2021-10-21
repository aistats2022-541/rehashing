import h5py
import os
import sys

def get_dataset_fn(dataset):
    if not os.path.exists('resources/data'):
        os.mkdir('resources/data')
    if not os.path.exists('resources/queries'):
        os.mkdir('resources/queries')
    if not os.path.exists('resources/exact'):
        os.mkdir('resources/exact')
    return (os.path.join('resources', 'data', '%s.txt' % dataset),
            os.path.join('resources', 'queries', '%s.txt' % dataset),
            os.path.join('resources', 'data', '%s.conf' % dataset),
            os.path.join('resources', 'exact', '%s_gaussian.txt' % dataset),
            )


def write_data(name, data, queries, ground_truth):
    data_fn, query_fn, _, groundtruth_fn = get_dataset_fn(name)
    print(get_dataset_fn(name))
    with open(data_fn, 'w') as f:
        for i, v in enumerate(data):
            f.write(str(i) + "," + ",".join(map(str, v)) + "\n")
    with open(query_fn, 'w') as f:
        for i, v in enumerate(queries):
            f.write(str(i) + "," + ",".join(map(str, v)) + "\n")
    with open(groundtruth_fn, 'w') as f:
        for i, val in enumerate(ground_truth):
            f.write(f'{val},{i}\n')


def write_config(name, n, d, m, bw):
    data_fn, query_fn, config_fn, groundtruth_fn = get_dataset_fn(name)
    print(f'writing config to {config_fn}')
    f = open(config_fn, 'w')

    config = """
    gaussian {
        name = "%s";
        fpath = "%s";
        qpath = "%s";
        exact_path = "%s";
        kernel = "gaussian";
        n = "%d";
        d = "%d";
        m = "%d";
        h = "%d";
        bw_const = "true";
        ignore_header = "false";
        start_col = "1";
        end_col = "%d";
        samples = "100";
        sample_ratio = "2.5";
        eps = "0.5";
        tau = "0.000001";
        beta = "0.5";
    }
    """ % (name, data_fn, query_fn, groundtruth_fn,
            n, d, m, bw, d)
    f.write(config)
    f.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <file>")
        exit(1)

    f = h5py.File(sys.argv[1], 'r')
    name = sys.argv[1].split("/")[-1][:-5]

    # only try for kde 0.01 in the beginning
    _, bw = f.attrs['kde.01']
    n = f['data'].shape[0]
    d = f['data'].shape[1]
    m = f['query'].shape[0]
    write_data(name, f['data'], f['query'], f['kde.01'])
    write_config(name, n, d, m, bw)











