# This code is adapted from github.com/erikbern/ann-benchmarks.

import numpy
import os
import random
import sys
try:
        from urllib import urlretrieve
        from urllib import urlopen
except ImportError:
        from urllib.request import urlretrieve
        from urllib.request import urlopen

def download(src, dst):
    if not os.path.exists(dst):
        # TODO: should be atomic
        print('downloading %s -> %s...' % (src, dst))
        urlretrieve(src, dst)


def get_dataset_fn(dataset):
    if not os.path.exists('resources/data'):
        os.mkdir('resources/data')
    return (os.path.join('resources', 'data', '%s.txt' % dataset),
            os.path.join('resources', 'data', '%s.conf' % dataset))

def get_exact_fn(dataset):
    if not os.path.exists('resources/exact'):
        os.mkdir('resources/exact')
    return os.path.join('resources', 'exact', '%s_gaussian.txt' % dataset)

def get_dataset(which):
    hdf5_fn = get_dataset_fn(which)
    try:
        url = 'http://todo/%s.hdf5' % which
        download(url, hdf5_fn)
    except:
        print("Cannot download %s" % url)
        if which in DATASETS:
            print("Creating dataset locally")
            DATASETS[which](hdf5_fn)
    hdf5_f = h5py.File(hdf5_fn, 'r')
    return hdf5_f


# Everything below this line is related to creating datasets

def write_output(X, fn, config_fn, queries=10000):
    from scipy import stats
    n = 0
    f = open(fn, 'w')
    print('test size:  %9d * %4d' % X.shape)
    Y = stats.zscore(X, axis=0)
    for i, v in enumerate(Y):
        f.write(str(i) + "," + ",".join(map(str, v)) + "\n")
    f.close()
    f = open(config_fn, 'w')
    dataset_name = fn.split("/")[-1]

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
        h = "2";
        bw_const = "false";
        ignore_header = "false";
        start_col = "1";
        end_col = "%d";
        samples = "100";
        sample_ratio = "2.5";
        eps = "0.5";
        tau = "0.0001";
        beta = "0.5";
    }
    """ % (dataset_name, fn, fn, get_exact_fn(dataset_name),
            X.shape[0], X.shape[1] - 1, queries, X.shape[1] - 1)
    f.write(config)
    f.close()
    print("finding bruteforce values")
    cmd = 'bin/hbe_exact ' +  config_fn + " gaussian"
    print(cmd)
    os.system('bash install.sh')
    os.system(cmd)


def covtype(out_fn, config_fn):
    import gzip
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
    fn = os.path.join('data', 'covtype.gz')
    download(url, fn)
    X = []
    with gzip.open(fn, 'rt') as t:
        for line in t.readlines():
            X.append([int(x) for x in line.strip().split(",")])
    write_output(numpy.array(X), out_fn, config_fn)

def shuttle(out_fn, config_fn):
    import zipfile
    X = []
    for dn in ("shuttle.trn.Z", "shuttle.tst"):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/shuttle/%s" % dn
        fn = os.path.join("data", dn)
        download(url, fn)
        if fn.endswith(".Z"):
            os.system("uncompress " + fn)
            fn = fn[:-2]
        with open(fn) as f:
            for line in f:
                X.append([int(x) for x in line.split()])
    write_output(numpy.array(X), out_fn, config_fn)

def glove(out_fn, config_fn, d):
    import zipfile

    url = 'http://nlp.stanford.edu/data/glove.twitter.27B.zip'
    fn = os.path.join('data', 'glove.twitter.27B.zip')
    download(url, fn)
    with zipfile.ZipFile(fn) as z:
        print('preparing %s' % out_fn)
        z_fn = 'glove.twitter.27B.%dd.txt' % d
        X = []
        for line in z.open(z_fn):
            v = [float(x) for x in line.strip().split()[1:]]
            X.append(numpy.array(v))
        write_output(numpy.array(X), out_fn, config_fn)

def _load_texmex_vectors(f, n, k):
    import struct

    v = numpy.zeros((n, k))
    for i in range(n):
        f.read(4)  # ignore vec length
        v[i] = struct.unpack('f' * k, f.read(k * 4))

    return v


def _get_irisa_matrix(t, fn):
    import struct
    m = t.getmember(fn)
    f = t.extractfile(m)
    k, = struct.unpack('i', f.read(4))
    n = m.size // (4 + 4 * k)
    f.seek(0)
    return _load_texmex_vectors(f, n, k)


def sift(out_fn, config_fn):
    import tarfile

    url = 'ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz'
    fn = os.path.join('data', 'sift.tar.tz')
    download(url, fn)
    with tarfile.open(fn, 'r:gz') as t:
        train = _get_irisa_matrix(t, 'sift/sift_base.fvecs')
        write_output(train, out_fn, config_fn)

def svhn(out_fn, config_fn, version):
    from scipy.io import loadmat
    url = 'http://ufldl.stanford.edu/housenumbers/%s_32x32.mat' % version
    fn = os.path.join('data', 'svhn-%s.mat' % version)
    download(url, fn)
    X = loadmat(fn)['X']
    d = numpy.prod(X.shape[:3])
    Y = numpy.reshape(X, (d, X.shape[3])).T
    write_output(Y, out_fn, config_fn, 500)

def _load_mnist_vectors(fn):
    import gzip
    import struct

    print('parsing vectors in %s...' % fn)
    f = gzip.open(fn)
    type_code_info = {
        0x08: (1, "!B"),
        0x09: (1, "!b"),
        0x0B: (2, "!H"),
        0x0C: (4, "!I"),
        0x0D: (4, "!f"),
        0x0E: (8, "!d")
    }
    magic, type_code, dim_count = struct.unpack("!hBB", f.read(4))
    assert magic == 0
    assert type_code in type_code_info

    dimensions = [struct.unpack("!I", f.read(4))[0]
                  for i in range(dim_count)]

    entry_count = dimensions[0]
    entry_size = numpy.product(dimensions[1:])

    b, format_string = type_code_info[type_code]
    vectors = []
    for i in range(entry_count):
        vectors.append([struct.unpack(format_string, f.read(b))[0]
                        for j in range(entry_size)])
    return numpy.array(vectors)


def mnist(out_fn, config_fn):
    download(
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', 'mnist-train.gz')  # noqa
    download(
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', 'mnist-test.gz')  # noqa
    train = _load_mnist_vectors('mnist-train.gz')
    write_output(train, out_fn, config_fn)


def fashion_mnist(out_fn, config_fn):
    download('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',  # noqa
             'fashion-mnist-train.gz')
    download('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',  # noqa
             'fashion-mnist-test.gz')
    train = _load_mnist_vectors('fashion-mnist-train.gz')
    test = _load_mnist_vectors('fashion-mnist-test.gz')
    write_output(train, out_fn, config_fn)

def random_float(out_fn, config_fn, n_dims, n_samples, centers, distance):
    import sklearn.datasets

    X, _ = sklearn.datasets.make_blobs(
        n_samples=n_samples, n_features=n_dims,
        centers=centers, random_state=1)
    write_output(X, out_fn, config_fn)

def lastfm(out_fn, config_fn, n_dimensions, test_size=50000):
    # This tests out ANN methods for retrieval on simple matrix factorization
    # based recommendation algorithms. The idea being that the query/test
    # vectors are user factors and the train set are item factors from
    # the matrix factorization model.

    # Since the predictor is a dot product, we transform the factors first
    # as described in this
    # paper: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/XboxInnerProduct.pdf  # noqa
    # This hopefully replicates the experiments done in this post:
    # http://www.benfrederickson.com/approximate-nearest-neighbours-for-recommender-systems/  # noqa

    # The dataset is from "Last.fm Dataset - 360K users":
    # http://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/lastfm-360K.html  # noqa

    # This requires the implicit package to generate the factors
    # (on my desktop/gpu this only takes 4-5 seconds to train - but
    # could take 1-2 minutes on a laptop)
    from implicit.datasets.lastfm import get_lastfm
    from implicit.approximate_als import augment_inner_product_matrix
    import implicit

    # train an als model on the lastfm data
    _, _, play_counts = get_lastfm()
    model = implicit.als.AlternatingLeastSquares(factors=n_dimensions)
    model.fit(implicit.nearest_neighbours.bm25_weight(
        play_counts, K1=100, B=0.8))

    # transform item factors so that each one has the same norm,
    # and transform the user factors such by appending a 0 column
    _, item_factors = augment_inner_product_matrix(model.item_factors)
    user_factors = numpy.append(model.user_factors,
                                numpy.zeros((model.user_factors.shape[0], 1)),
                                axis=1)

    # only query the first 50k users (speeds things up signficantly
    # without changing results)
    user_factors = user_factors[:test_size]

    # after that transformation a cosine lookup will return the same results
    # as the inner product on the untransformed data
    write_output(item_factors, out_fn, config_fn)


DATASETS = {
    'fashion-mnist-784-euclidean': fashion_mnist,
    'glove-100-angular': lambda out_fn: glove(out_fn, config_fn, 100),
    'mnist-784-euclidean': mnist,
    'random-xs-20-euclidean': lambda out_fn: random_float(out_fn, config_fn, 20, 10000, 100,
                                                    'euclidean'),
    'random-s-100-euclidean': lambda out_fn: random_float(out_fn, config_fn, 100, 100000, 1000,
                                                    'euclidean'),
    'sift-128-euclidean': sift,
    'lastfm-64-dot': lambda out_fn: lastfm(out_fn, config_fn, 64),
    'covtype': lambda out_fn, config_fn: covtype(out_fn, config_fn),
    'shuttle': lambda out_fn, config_fn: shuttle(out_fn, config_fn),
    'svhn': lambda out_fn, config_fn: svhn(out_fn, config_fn, 'extra'),
    'svhn-small': lambda out_fn, config_fn: svhn(out_fn, config_fn, 'test'),
}

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        choices=DATASETS.keys(),
        required=True)
    args = parser.parse_args()
    fn, gn = get_dataset_fn(args.dataset)
    DATASETS[args.dataset](fn, gn)
