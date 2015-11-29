import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from load import mnist
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
theano.config.floatX = 'float32'
srng = RandomStreams()
import Plots

f = open("costs.txt", 'w')
f.write("Starting...\n")
f.close()

def write(str):
    f = open("costs.txt", 'a')
    f.write(str)
    f.close()

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def rectify(X):
    return T.maximum(X, 0.)

def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

def model(X, filter_params, fc_params, p_drop_conv, p_drop_hidden):
    inp = X
    outshp = 28
    params = []
    for f in filter_params:
        outa = rectify(conv2d(inp, f, border_mode='valid'))
        outb = max_pool_2d(outa, (2, 2))
        outc = dropout(outb, p_drop_conv)

        f_shp = f.get_value().shape
        outshp = (outshp - f_shp[2] + 1)/2
        inp = outc

    inp = T.flatten(inp, outdim=2)
    outshp = filter_params[-1].get_value().shape[0] * outshp * outshp
    for w in fc_params[:-1]:
        out = rectify(T.dot(inp, w))
        out = dropout(out, p_drop_hidden)
        inp = out
    w = fc_params[-1]
    pyx = softmax(T.dot(out, w))
    return pyx

def get_params(filters, fc):
    outshp = 28
    filter_params = []
    fc_params = []
    for f in filters:
        w = init_weights(f)
        filter_params.append(w)
        outshp = (outshp - f[2] + 1)/2

    outshp = filters[-1][0] * outshp * outshp
    w = init_weights((outshp, fc[0]))
    fc_params.append(w)

    for i in range(len(fc)-1):
        w = init_weights((fc[i], fc[i+1]))
        fc_params.append(w)
    return filter_params, fc_params




trX, teX, trY, teY = mnist(onehot=True)
print(trX.shape)
print(trY.shape)
img_x = 28
img_y = 28

trX = trX.reshape(-1, 1, img_x, img_y)
teX = teX.reshape(-1, 1, img_x, img_y)

X = T.ftensor4()
Y = T.fmatrix()

f1 = (64, 1, 11, 11)
f2 = (100, f1[0], 4, 4)
filters = [f1, f2]
fc = [600, 200, 10]

filter_params, fc_params = get_params(filters, fc)
params = filter_params + fc_params
print(params)
noise_py_x = model(X, filter_params, fc_params, 0.5, 0.5)

py_x = model(X, filter_params, fc_params, 0.0, 0.0)
y_x = T.argmax(py_x, axis=1)


cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
updates = RMSprop(cost, params, lr=0.001)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

for i in range(100):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        cost = train(trX[start:end], trY[start:end])
        print(cost)
        # write(str(i) + ": " + str(start) + ": " + str(cost) + "\n")
    Plots.plot_filters(params[0].get_value(), i, "")
    print(np.mean(np.argmax(teY, axis=1) == predict(teX)))
    write(str(i) + ": " + str(np.mean(np.argmax(teY, axis=1) == predict(teX))))
    write("\n")

