import numpy as np
import os
import gzip
import pickle

datasets_dir = 'data-mnist/'
datasets_dir = "data-mnist/mnist.pkl.gz"
def one_hot(x,n):
	if type(x) == list:
		x = np.array(x)
	x = x.flatten()
	o_h = np.zeros((len(x),n))
	o_h[np.arange(len(x)),x] = 1
	return o_h
def mnist(ntrain=60000,ntest=10000,onehot=True):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''
    dataset = datasets_dir
    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(os.path.split(__file__)[0],
                                "..", "data", dataset)
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print('Downloading data from %s' % origin)
        urllib.urlretrieve(origin, dataset)

    print('... loading data')

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    f.close()
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is an numpy.ndarray of 2 dimensions (a matrix)
    # which row's correspond to an example. target is a
    # numpy.ndarray of 1 dimensions (vector)) that have the same length as
    # the number of rows in the input. It should give the target
    # target to the example with the same index in the input.
    print("Done.")
    #return (train_set, valid_set, test_set)
    if onehot:
        trY = one_hot(train_set[1], 10)
        teY = one_hot(valid_set[1], 10)
    else:
        trY = np.asarray(train_set[1])
        teY = np.asarray(valid_set[1])
    trX = train_set[0]
    teX = valid_set[0]
    print(trX.shape)
    return trX,teX,trY,teY


