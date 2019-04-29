# self module
from .optim import *
from .layers import *

# for io task
import pickle as pickle



class FullyConnectedNet:

    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU activation, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,

    iteration in hidden layer
      affine -> batch norm-> relu -> dropout
    output layer
      affine -> softmax_loss
    """


    def __init__(self, hidden_dims=[], input_dim=32*32*3, num_class=10, use_bacthnorm=False, dropout=0, reg=0, checkpoint_name=None,
                 weight_scale=1e-2, learning_rate_init=1e-5, solver=Adam, lr_decay=1, batch_size=64, verbose=False, num_epoch=10):

        """
        Initialize a new FullyConnectedNet.
            Inputs:
            - hidden_dims: A list of integers giving the size of each hidden layer.
            - input_dim: An integer giving the size of the input.
            - num_classes: An integer giving the number of classes to classify.
            - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 not use dropout
            - use_batchnorm: Whether or not the network should use batch normalization.
            - reg: Scalar giving L2 regularization strength.
            - weight_scale: Scalar giving the standard deviation for random init of the weights.
            - solver: a giving the name of an update rule in optim.py
            - learning_rate_init: learning rate in Initialize
            - optim_config: A dictionary containing hyper-parameters that will be
                passed to the chosen update rule. Each update rule requires different
                hyper-parameters (see optim.py)
            - lr_decay: A scalar for learning rate decay; after each epoch the
                learning rate is multiplied by this value.
            - batch_size: Size of mini-batches used to compute loss and gradient during training.
            - verbose: if set to false then no output will be printed during training.
            - checkpoint_name: If not None, then save model checkpoints here every epoch.

        Exam:
            - 3 hidden layers and in each layer 100 neuron
        >>> clf = FullyConnectedNet(hidden_dims=[100, 100, 100], num_class=10)
        >>> clf.train((x, y), (x, y))

        """

        self.num_dims = 1 + len(hidden_dims)
        self.dims = [input_dim] + hidden_dims + [num_class]
        self.reg = reg
        self.solver = solver
        self.lr = learning_rate_init
        self.weight_scale = weight_scale
        self.batchnorm = use_bacthnorm
        self.lr_decay = lr_decay
        self.batch_size = batch_size
        self.verbose = verbose
        self.epoch = num_epoch
        self.checkpoint_name = checkpoint_name
        self.p_drop = dropout

        # all parameters of FullyConnected model
        self.param_, self.optim_config_ = self.create_params()
        # number of epochs is trained
        self.trained_epoch = 0
        # list of loss each iter
        self.loss_history = []



    def create_params(self):
        param = {}
        config = {}
        for i in range(len(self.dims) - 1):
            param['w%d' % i] = np.random.randn(self.dims[i], self.dims[i + 1]) * self.weight_scale
            param['b%d' % i] = np.zeros(self.dims[i + 1])
            config['w%d' % i] = 0
            config['b%d' % i] = 0
            if i < len(self.dims) - 2 and self.batchnorm:
                param['gamma%d' % i] = np.ones(self.dims[i + 1])
                param['beta%d' % i] = np.zeros(self.dims[i + 1])
                param['run_guss%d' % i] = {}
                config['gamma%d' % i] = 0
                config['beta%d' % i] = 0
        return param, config



    def forward_step(self, param, h):
        # forward step with (dropout, batchnorm)
        caches = {'h0': h}
        score = 0

        for i in range(self.num_dims):
            if i < self.num_dims - 1:
                h = liner_forward(h, param['w%d' % i], param['b%d' % i])
                if self.batchnorm:
                    h, caches['bn%d' % i] = batchnorm_forward(h, param['gamma%d' % i], param['beta%d' % i], param['run_guss%d' % i])
                    caches['norm%d' % (i + 1)] = h
                h, caches['mask%d' % (i + 1)] = dropout_forward(relu(h), self.p_drop)
                caches['h%d' % (i + 1)] = h
            else:
                score = liner_forward(h, param['w%d' % i], param['b%d' % i])

        return score, caches



    def backward_step(self, dout, param, cache):
        # backward step with (dropout, batchnorm)
        grads = {}
        for i in range(self.num_dims - 1, -1, -1):
            if i == self.num_dims - 1:
                dh, grads['w%d' % i], grads['b%d' % i] = liner_backward(dout, cache['h%d' % i], param['w%d' % i])
            else:
                dmask = dropout_backward(dh, cache['mask%d' % (i + 1)])
                if self.batchnorm:
                    dmax = max_backward(dmask, cache['norm%d' % (i + 1)])
                    dnorm, grads['gamma%d' % i], grads['beta%d' % i] = batchnorm_backward(dmax, cache['bn%d' % i])
                    dh, grads['w%d' % i], grads['b%d' % i] = liner_backward(dnorm, cache['h%d' % i], param['w%d' % i])
                else:
                    dmax = max_backward(dmask, cache['h%d' % (i + 1)])
                    dh, grads['w%d' % i], grads['b%d' % i] = liner_backward(dmax, cache['h%d' % i], param['w%d' % i])
        return grads


    def forward_test(self, param, h):
        score = 0
        for i in range(self.num_dims):
            if i < self.num_dims - 1:
                h = h @ param['w%d' % i] + param['b%d' % i]
                if self.batchnorm:
                    h, _ = batchnorm_forward(h, param['gamma%d' % i], param['beta%d' % i], param['run_guss%d' % i], mode='test')
                h = relu(h)
            else:
                score = liner_forward(h, param['w%d' % i], param['b%d' % i])
        return score



    def update_param(self, param, grads, config):
        for i in range(self.num_dims):
            # added reg
            grads['w%d' % i] += self.reg * param['w%d' % i]
            # update_param
            param['w%d' % i], config['w%d' % i] = self.solver(param['w%d' % i], grads['w%d' % i], self.lr, config['w%d' % i])
            param['b%d' % i], config['b%d' % i] = self.solver(param['b%d' % i], grads['b%d' % i], self.lr, config['b%d' % i])
            if i < self.num_dims - 1 and self.batchnorm:
                param['gamma%d'%i], config['gamma%d'%i] = self.solver(param['gamma%d'%i], grads['gamma%d'%i], self.lr, config['gamma%d'%i])
                param['beta%d'%i], config['beta%d'%i] = self.solver(param['beta%d'%i], grads['beta%d'%i], self.lr, config['beta%d'%i])

        return param, config




    def save_model(self, path):
        if self.checkpoint_name is None: return
        checkpoint = {'model': self}

        filename = '%s_epoch_%d.pkl' % (path, self.epoch)
        print('\t Saving model to "%s"' % path)
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)


    @staticmethod
    def load_model(filename):
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        return model



    def predict(self, x_test):
        score = self.forward_test(self.param_, x_test)
        return np.argmax(score, axis=1)


    def score(self, x_test, y_test):
        y_pred = self.predict(x_test)
        return np.mean(y_pred == y_test)


    def evaluate(self, x_test, y_test, batch=5):
        # evaluate model for after training
        scores = []
        size = x_test.shape[0] // batch
        for i in  range(batch):
            x_batch, y_batch = mini_batch(x_test, y_test, size)
            y_pred = self.predict(x_batch)
            score = np.mean(y_pred == y_batch)
            print('\t score in batch %d  %.2f'%(i, score))
            scores.append(score)
        print('   mean score is %.2f'%np.mean(scores))


    def compute_ephoce(self, num_data):
        iter_per_epoch = max(num_data // self.batch_size, 1)
        num_iter = iter_per_epoch * self.epoch
        return num_iter, iter_per_epoch



    def train(self, train_data=(), val_data=()):
        x_train, y_train = train_data
        x_val, y_val = val_data
        num_iter, iter_per_epoch = self.compute_ephoce(x_train.shape[0])

        for i in range(num_iter + 1):
            # forward and compute loss
            x_batch, y_batch = mini_batch(x_train, y_train, self.batch_size)
            scores, cache = self.forward_step(self.param_, x_batch)
            loss, dout = softmax_loss(scores, y_batch, self.param_, self.reg)
            self.loss_history.append(loss)

            # backward and update param
            grads = self.backward_step(dout, self.param_, cache)
            self.param_, self.optim_config_= self.update_param(self.param_, grads, self.optim_config_)

            if i%iter_per_epoch == 0 and self.verbose:
                self.lr *= self.lr_decay
                train_acc = self.score(x_train, y_train)
                val_acc = self.score(x_val, y_val)

                print('\t\t epchoe %4d | loss: %.4f | x_train: %.3f | x_val: %.3f' % (self.trained_epoch, loss, train_acc, val_acc))
                self.save_model(self.checkpoint_name)
                self.trained_epoch += 1

        return