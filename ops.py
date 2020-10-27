import os
import tensorflow as tf
import numpy as np

def config_common():
    gpu_option = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(allow_soft_placement = True, gpu_options=gpu_option)
    return config

def total_params(g):
    print('# of parameter : %e' % np.sum([np.product(node.shape)for node in g.get_collection('trainable_variables')]).value)


def dlist(dname):
    return [os.path.join(dname, fname) for fname in os.listdir(dname) if os.path.isfile(os.path.join(dname, fname))]


def mkdir(dname):
    if not os.path.exists(dname):
        os.makedirs(dname)


def selfAttention(img, scope='self_attention'):
    with tf.variable_scope(scope):
        b, w, h, c_in = img.shape
        init = tf.random_normal_initializer(0.0, 0.02)

        key = tf.layers.conv2d(img, c_in//8, 1, 1, padding='valid', kernel_initializer=init)
        query = tf.layers.conv2d(img, c_in//8, 1, 1, padding='valid', kernel_initializer=init)
        value = tf.layers.conv2d(img, c_in, 1, 1, padding='valid', kernel_initializer=init)

        key_flatten = tf.reshape(key, [tf.shape(img)[0], -1, c_in//8])
        query_flatten = tf.reshape(query, [tf.shape(img)[0], -1, c_in//8])
        value_flatten = tf.reshape(value, [tf.shape(img)[0], -1, c_in])

        beta = tf.nn.softmax(tf.matmul(query_flatten, key_flatten, transpose_b=True))
        feature_flatten = tf.matmul(beta, value_flatten)
        feature_map = tf.reshape(feature_flatten, [tf.shape(img)[0], tf.shape(img)[1], tf.shape(img)[2], c_in])

        gamma = tf.get_variable('gamma_attention', [1], initializer=tf.constant_initializer(0.9))
        feature_map = img + gamma * feature_map

    return feature_map

def conv2d_ws(img, c, k, s, norm=None, activation=None, is_train=True, padding='SAME', scope='scope'):
    _, _, _, c_in = img.shape
    dtype = tf.float32
    shape = [k, k, c_in, c]
    init = tf.random_normal_initializer(0.0, 0.02)
    eps = 1e-6
    stride = [1, s, s, 1]
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        kernel = tf.get_variable('kernel', shape, dtype, init)
        bias = tf.get_variable('bias', [], dtype, tf.zeros_initializer())
        mean, var = tf.nn.moments(kernel, [0, 1, 2])
        kernel_normalized = (kernel - mean) / (tf.sqrt(var) + eps)
        
        net = tf.nn.conv2d(img, kernel_normalized, stride, padding)
        net += bias
        net = normalize(net, norm, is_train)
        net = activate(net, activation)

    return net

def deconv2d_ws(img, c, k, s, norm=None, activation=None, is_train=True, padding='SAME', scope='scope'):
    _, _, _, c_in = img.shape
    dtype = tf.float32
    shape = [k, k, c, c_in]
    output_shape = [tf.shape(img)[0], img.get_shape()[1] * 2, img.get_shape()[2] * 2, c]
    init = tf.random_normal_initializer(0.0, 0.02)
    eps = 1e-6
    stride = [1, s, s, 1]
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        kernel = tf.get_variable('kernel', shape, dtype, init)
        bias = tf.get_variable('bias', [], dtype, tf.zeros_initializer())
        mean, var = tf.nn.moments(kernel, [0, 1, 2])
        kernel_normalized = (kernel - mean) / (tf.sqrt(var) + eps)
        
        net = tf.nn.conv2d_transpose(img, kernel_normalized, output_shape, stride, padding)
        net += bias
        net = normalize(net, norm, is_train)
        net = activate(net, activation)

    return net

def gatedConv(img, c, k, s, d=1, norm=None, activation=None, is_train=True, scope='conv'):
    init = tf.random_normal_initializer(0.0, 0.02)
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        gating = tf.layers.conv2d(img, c, k, s, padding='same', dilation_rate=d, kernel_initializer=init)
        feature = tf.layers.conv2d(img, c, k, s, padding='same', dilation_rate=d, kernel_initializer=init)
        feature = normalize(feature, norm, is_train)
        feature = activate(feature, activation)

        gatedFeature = tf.multiply(feature, tf.sigmoid(gating))
        return gatedFeature

def gatedDeconv(img, c, k, s, norm=None, activation=None, is_train=True, scope='deconv'):
    init = tf.random_normal_initializer(0.0, 0.02)
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        gating = tf.layers.conv2d_transpose(img, c, k, s, padding='same', kernel_initializer=init)
        feature = tf.layers.conv2d_transpose(img, c, k, s, padding='same', kernel_initializer=init)
        feature = normalize(feature, norm, is_train)
        feature = activate(feature, activation)

        gatedFeature = tf.multiply(feature, tf.sigmoid(gating))
        return gatedFeature


def partial_conv2d(X, c, k, s, sn=False):
    b, h, w, c_in = X.shape
    init = tf.random_normal_initializer(0.0, 0.02)

    if sn:
        w_sn = tf.get_variable("spectral_norm_kernel", shape=[k, k, X.shape[-1], c], initializer=init)
        bias = tf.get_variable("parital_bias", shape=[], dtype=tf.float32, initializer=tf.zeros_initializer())
        X_p0 = tf.nn.conv2d(X, spectral_norm(w_sn), [1, s, s, 1], 'SAME')
    else:
        bias = tf.get_variable("bias", shape=[], dtype=tf.float32, initializer=tf.zeros_initializer())
        X_p0 = tf.layers.conv2d(X, c, k, s, 'same', use_bias=False, kernel_initializer=init)

    # scale-factor calculation
    mask = tf.constant(1.0, tf.float32, [1, h, w, 1])
    p0_filter = tf.constant(1., tf.float32, [k, k, 1, 1])
    scale = tf.nn.conv2d(mask, p0_filter, [1, s, s, 1], 'SAME')

    out = X_p0 * (1. / (scale / k ** 2)) + bias
    return out


def conv2d(x, c, k, s, norm=None, activation=None, is_train=True, partial=False, sn=False, scope='conv'):
    init = tf.random_normal_initializer(0.0, 0.02)
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if partial and sn:
            net = partial_conv2d(x, c, k, s, sn=True)
            #raise ValueError("Can't use parital and spectral norm simultaneously")

        elif partial:
            net = partial_conv2d(x, c, k, s)

        elif sn:
            p = int((k-1)/2)
            net = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], 'SAME')

            w = tf.get_variable("kernel", shape=[k, k, x.shape[-1], c], initializer=init)
            net = tf.nn.conv2d(net, spectral_norm(w), [1, s, s, 1], 'VALID')

        else:
            # net = x
            # p = int((k-1)/2)
            # net = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], 'REFLECT')
            net = tf.layers.conv2d(x, c, k, s, 'same', kernel_initializer=init)

        net = normalize(net, norm, is_train)
        net = activate(net, activation)
        return net


def deconv2d(x, c, k, s, norm=None, activation=None, is_train=True, sn=False, scope='deconv'):
    b, _, _, _ = x.get_shape().as_list()

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        init = tf.random_normal_initializer(0.0, 0.02)

        if sn:
            w = tf.get_variable("kernel", shape=[k, k, c, x.shape[-1]], initializer=init)
            net = tf.nn.conv2d_transpose(x, spectral_norm(w), [b, int(x.shape[1] * 2), int(x.shape[2] * 2), c], [1, s, s, 1], 'SAME')
            # net = tf.nn.conv2d_transpose(x, spectral_norm(w), [1, int(output_size), int(output_size), c], [1, s, s, 1], 'SAME')

        else:
            net = tf.layers.conv2d_transpose(x, c, k, s, 'same', kernel_initializer=init)

        net = normalize(net, norm, is_train)
        net = activate(net, activation)
        return net


def flatten(x):
    return tf.layers.flatten(x)


def residual(x, c, norm, is_train=True, partial=False, sn=False, scope='res'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        c_in = x.shape[-1]
        net = conv2d(x, c, 3, 1, norm, 'relu', is_train, partial, sn, scope=scope+'_01')
        net = conv2d(net, c, 3, 1, norm=norm, is_train=is_train, partial=partial, sn=sn, scope=scope+'_02')

        if c_in != c:
            sc = conv2d(x, c, 1, 1, is_train=is_train, partial=partial, sn=sn, scope=scope+'_sc')
            return activate(sc + net, 'relu')
        else:
            return activate(x + net, 'relu')


def residual_ws(x, c, norm, is_train=True, scope='res'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        c_in = x.shape[-1]
        net = conv2d_ws(x, c, 3, 1, norm, 'relu', is_train, scope=scope+'_01')
        net = conv2d_ws(net, c, 3, 1, norm=norm, is_train=is_train, scope=scope+'_02')

        if c_in != c:
            sc = conv2d(x, c, 1, 1, is_train=is_train, scope=scope+'_sc')
            return activate(sc + net, 'relu')
        else:
            return activate(x + net, 'relu')


def resblock(x, norm, activation, is_train):
    c = x.shape[-1].value
    net = conv2d(x, c, 3, 1, norm, activation, is_train)
    net = conv2d(net, c, 3, 1, norm, activation, is_train) 
    
    return x + net


def mlp(x, dim, norm=None, activation=None, is_train=True):
    init = tf.random_normal_initializer(0.0, 0.02)
    net = tf.layers.dense(x, dim, kernel_initializer=init)
    net = normalize(net, norm, is_train)
    net = activate(net, activation)
    return net


def activate(x, activation=None):
    if activation == 'relu':
        return tf.nn.relu(x)
    elif activation == 'sigmoid':
        return tf.sigmoid(x)
    elif activation == 'tanh':
        return tf.tanh(x)
    elif activation == 'leaky':
        return tf.nn.leaky_relu(x)
    elif activation == None:
        return x
    elif activation == 'elu':
        return tf.nn.elu(x)
    else:
        raise NameError('Invalid activation')


def normalize(x, norm, is_train):
    init = tf.random_normal_initializer(0.0, 0.02)
    if norm == 'batch':
        # return tf.layers.batch_normalization(x, axis=3, epsilon=1e-5, momentum=0.1, training=is_train,
        #                                      gamma_initializer=tf.random_normal_initializer(1.0, 0.02))
        return tf.layers.batch_normalization(x, training=is_train)
    elif norm == 'instance':
        return tf.contrib.layers.instance_norm(x)
    elif norm == 'bin':
        return bin(x, is_train)
    elif norm == None:
        return x
    elif norm == 'group':
        return tf.contrib.layers.group_norm(x, groups=16)
    else:
        raise NameError('Invalid normalization')


def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

        return w_norm



def bin(x, is_train):
    gamma = tf.Variable(0.1, dtype=tf.float32, name='bin_gamma')
    beta = tf.Variable(0, dtype=tf.float32, name='bin_beta')
    rho = tf.Variable(0.5, dtype=tf.float32, name='bin_rho')
    rho_clip = tf.clip_by_value(rho, 0, 1, name='bin_rho_clipped')

    batch_norm = tf.layers.batch_normalization(x, training=is_train)
    instance_norm = tf.contrib.layers.instance_norm(x)

    bin_norm = (rho_clip * batch_norm + (1-rho_clip) * instance_norm) * gamma + beta
    return bin_norm


def downsample(x):
    tmp = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    return tf.layers.average_pooling2d(tmp, [3, 3], 2, 'VALID')

def pooling(x, k, s, type='MAX'):
    if type == 'MAX':
        return tf.layers.max_pooling2d(x, [k, k], s, 'same')
    elif type == 'AVG':
        return tf.layers.average_pooling2d(x, [k, k], s, 'same')
    else:
        raise NameError("Invalid Pooling Type")
