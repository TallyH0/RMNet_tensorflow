import tensorflow as tf
import ops

def RMNet_block(x, c, is_train, dropout, activation, scope='RMNet_block', spatial_reduction=False):
    norm = 'batch'
    stride = 2 if spatial_reduction else 1
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        spatial = x
        if spatial_reduction:
            spatial = ops.pooling(x, 3, 2)
            spatial = ops.conv2d(spatial, c, 1, 1, norm=norm, is_train=is_train, scope='spatial_conv')
            
        net = ops.conv2d(x, c//4, 1, 1, norm=norm, activation=activation, is_train=is_train, scope='conv1')
        net = ops.conv2d(net, c//4, 3, stride, norm=norm, activation=activation, is_train=is_train, scope='conv2')
        net = ops.conv2d(net, c, 1, 1, norm=norm, activation=None, is_train=is_train, scope='conv3')
        net = tf.nn.dropout(net, dropout)
        
        net = ops.activate(spatial + net, activation)
        return net
    
def RMNet_backbone(x, n_c, is_train, dropout, activation, scope):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        net = ops.conv2d(x, n_c, 3, 2, is_train=is_train, scope='Block1_00')
        net = RMNet_block(net, n_c, is_train, dropout, activation, 'Block1_01', spatial_reduction=False)
        net = RMNet_block(net, n_c, is_train, dropout, activation, 'Block1_02', spatial_reduction=False)
        net = RMNet_block(net, n_c, is_train, dropout, activation, 'Block1_03', spatial_reduction=False)
        net = RMNet_block(net, n_c, is_train, dropout, activation, 'Block1_04', spatial_reduction=False)
        
        net = RMNet_block(net, n_c*2, is_train, dropout, activation, 'Block2_00', spatial_reduction=True)
        net = RMNet_block(net, n_c*2, is_train, dropout, activation, 'Block2_01', spatial_reduction=False)
        net = RMNet_block(net, n_c*2, is_train, dropout, activation, 'Block2_02', spatial_reduction=False)
        net = RMNet_block(net, n_c*2, is_train, dropout, activation, 'Block2_03', spatial_reduction=False)
        net = RMNet_block(net, n_c*2, is_train, dropout, activation, 'Block2_04', spatial_reduction=False)
        net = RMNet_block(net, n_c*2, is_train, dropout, activation, 'Block2_05', spatial_reduction=False)
        net = RMNet_block(net, n_c*2, is_train, dropout, activation, 'Block2_06', spatial_reduction=False)
        net = RMNet_block(net, n_c*2, is_train, dropout, activation, 'Block2_07', spatial_reduction=False)
        net = RMNet_block(net, n_c*2, is_train, dropout, activation, 'Block2_08', spatial_reduction=False)
    
        net = RMNet_block(net, n_c*4, is_train, dropout, activation, 'Block3_00', spatial_reduction=True)
        net = RMNet_block(net, n_c*4, is_train, dropout, activation, 'Block3_01', spatial_reduction=False)
        net = RMNet_block(net, n_c*4, is_train, dropout, activation, 'Block3_02', spatial_reduction=False)
        net = RMNet_block(net, n_c*4, is_train, dropout, activation, 'Block3_03', spatial_reduction=False)
        net = RMNet_block(net, n_c*4, is_train, dropout, activation, 'Block3_04', spatial_reduction=False)
        net = RMNet_block(net, n_c*4, is_train, dropout, activation, 'Block3_05', spatial_reduction=False)
        net = RMNet_block(net, n_c*4, is_train, dropout, activation, 'Block3_06', spatial_reduction=False)
        net = RMNet_block(net, n_c*4, is_train, dropout, activation, 'Block3_07', spatial_reduction=False)
        net = RMNet_block(net, n_c*4, is_train, dropout, activation, 'Block3_08', spatial_reduction=False)
        net = RMNet_block(net, n_c*4, is_train, dropout, activation, 'Block3_09', spatial_reduction=False)
        net = RMNet_block(net, n_c*4, is_train, dropout, activation, 'Block3_10', spatial_reduction=False)
    
        net = RMNet_block(net, n_c*8, is_train, dropout, activation, 'Block4_00', spatial_reduction=True)
        net = RMNet_block(net, n_c*8, is_train, dropout, activation, 'Block4_01', spatial_reduction=False)
        net = RMNet_block(net, n_c*8, is_train, dropout, activation, 'Block4_02', spatial_reduction=False)
        net = RMNet_block(net, n_c*8, is_train, dropout, activation, 'Block4_03', spatial_reduction=False)
        net = RMNet_block(net, n_c*8, is_train, dropout, activation, 'Block4_04', spatial_reduction=False)
        net = RMNet_block(net, n_c*8, is_train, dropout, activation, 'Block4_05', spatial_reduction=False)
        net = RMNet_block(net, n_c*8, is_train, dropout, activation, 'Block4_06', spatial_reduction=False)
        net = RMNet_block(net, n_c*8, is_train, dropout, activation, 'Block4_07', spatial_reduction=False)
        net = RMNet_block(net, n_c*8, is_train, dropout, activation, 'Block4_08', spatial_reduction=False)
        net = RMNet_block(net, n_c*8, is_train, dropout, activation, 'Block4_09', spatial_reduction=False)
        net = RMNet_block(net, n_c*8, is_train, dropout, activation, 'Block4_10', spatial_reduction=False)
        net = RMNet_block(net, n_c*8, is_train, dropout, activation, 'Block4_11', spatial_reduction=False)
        
        return net
    
def RMNet_ReID_head(x, activation, scope):
    b, h, w, c = x.shape
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        net = tf.reduce_max(x, axis=[1, 2], keepdims=True)
        net = ops.conv2d(net, c*2, 1, 1, activation=activation, scope='conv01')
        net = ops.conv2d(net, c, 1, 1, scope='conv02')
        net = tf.nn.l2_normalize(net, axis=3)
        return net
    
def RMNet_Calibration(x, activation, scope):
    b, h, w, c = x.shape
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        net = ops.conv2d(x, c, 1, 1, scope='conv01')
        net = tf.nn.l2_normalize(net, axis=3)
        return net
    
def RMNet(x, is_train, dropout, activation = 'elu', num_class = None):
    end_point = {
        'backbone' : None,
        'logit_local' : None,
        'logit_global' : None
    }
    t_backbone = RMNet_backbone(x, 32, is_train, dropout, activation, 'Backbone')
    end_point['backbone'] = t_backbone

    net_local = RMNet_ReID_head(t_backbone, activation, 'ReID_Head')
    embedding_local = tf.squeeze(net_local)
    if num_class:
        logit_local = tf.squeeze(ops.conv2d(net_local, num_class, 1, 1, scope='logit_local'))
        end_point['logit_local'] = logit_local
    
    net_global = RMNet_Calibration(net_local, activation, 'Calibration')
    embedding_global = tf.squeeze(net_global)
    if num_class:
        logit_global = tf.squeeze(ops.conv2d(net_global, num_class, 1, 1, scope='logit_global'))
        end_point['logit_global'] = logit_global
    
    return embedding_local, embedding_global, end_point
