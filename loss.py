import tensorflow as tf

def structure_loss(features, label, alfa, nrof_classes, scope='structure_loss'):
    """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        margin = 1
        # margin = tf.Variable(1., name='margin')
        nrof_batch = features.get_shape()[0]
        nrof_features = features.get_shape()[1]
        init_zero = tf.zeros_initializer()
        centers = tf.get_variable('centers', [nrof_classes, nrof_features], dtype=tf.float32,
            initializer=init_zero, trainable=False)
        label = tf.reshape(label, [-1])
        centers_batch = tf.gather(centers, label)
        diff = (1 - alfa) * (centers_batch - features)
        centers = tf.scatter_sub(centers, label, diff)
        with tf.control_dependencies([centers]):
            loss_center = tf.reduce_mean(tf.square(features - centers_batch))
            feature_norm = tf.reduce_sum(tf.square(features), axis=1, keepdims=True)
            center_norm = tf.reduce_sum(tf.square(centers), axis=1, keepdims=True)

            feature_norm_cast = tf.broadcast_to(feature_norm, [nrof_batch, nrof_batch])
            feature_norm_cast = feature_norm_cast + tf.transpose(feature_norm_cast, [1, 0])
            
            feature_dot_product = tf.matmul(features, features, transpose_b=True)
            feature_diff = feature_norm_cast - 2 * feature_dot_product
            
            loss_push = tf.nn.relu(-tf.reduce_mean(feature_diff) + loss_center + margin)

            feature_norm_cast2 = tf.broadcast_to(feature_norm, [nrof_batch, nrof_classes])
            center_norm_cast = tf.broadcast_to(center_norm, [nrof_classes, nrof_batch])
            
            feature_center_norm = feature_norm_cast2 + tf.transpose(center_norm_cast, [1, 0])
            feature_center_product = tf.matmul(features, centers, transpose_b=True)
            
            feature_center_diff = feature_center_norm - 2 * feature_center_product
            loss_gpush = tf.nn.relu(-tf.reduce_mean(feature_center_diff) + 2 * loss_center + margin)
        

        return loss_center, loss_push, loss_gpush, centers, margin
