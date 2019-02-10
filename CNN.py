import tensorflow as tf
# import tensorflow.contrib.layers as layers
from prepare_data import load_train_datatset, load_test_dataset, split_to_train_dev, batch

IMAGES_SHAPE = (48, 48, 1)
N_LABELS = 8
LR = 0.001


class CNN:
    def __init__(self, input_shape, n_labels):
        self.build_model(input_shape, n_labels)

    def build_model(self, input_shape, n_labels):
        self.input_ = tf.placeholder(tf.float32, (None,) + input_shape, name="input_")
        self.tf_y = tf.placeholder(tf.float32, (None, N_LABELS), name="Y")
        with tf.name_scope('convolution'):
            with tf.variable_scope("conv_" + str(0)):
                conv1 = tf.layers.conv2d(self.input_, 16, kernel_size=5, padding='SAME',
                                         activation=tf.nn.relu)
                max_pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=[2, 2])
                batch_norm1 = tf.layers.batch_normalization(max_pool1)
                dropout1 = tf.layers.dropout(batch_norm1, rate=0.5)

            with tf.variable_scope("conv_" + str(1)):
                conv2 = tf.layers.conv2d(dropout1, 32, kernel_size=5, padding='SAME',
                                         activation=tf.nn.relu)
                max_pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=[2, 2])
                batch_norm2 = tf.layers.batch_normalization(max_pool2)
                dropout2 = tf.layers.dropout(batch_norm2, rate=0.5)
        with tf.name_scope('fully_connected'):
            flat = tf.layers.flatten(dropout2)
            dense1 = tf.layers.dense(flat, 64)
            dropout1 = tf.layers.dropout(dense1, rate=0.5)
            dense2 = tf.layers.dense(dropout1, 64)
            dropout2 = tf.layers.dropout(dense2, rate=0.5)
        output = tf.layers.dense(dropout2, n_labels)
        with tf.name_scope('loss'):
            self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.tf_y, logits=output)
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(LR).minimize(self.loss)
        with tf.name_scope('accuracy'):
            self.accuracy = tf.metrics.accuracy(  # return (acc, update_op), and create 2 local variables
                labels=tf.argmax(self.tf_y, axis=1), predictions=tf.argmax(output, axis=1), )[1]

    def train(self, n_epoh=60, model_name=None):
        X, Y = load_train_datatset()
        # X_test, Y_test = load_test_dataset()
        X_train, Y_train, X_dev, Y_dev = split_to_train_dev(X, Y)
        sess = tf.Session()
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        saver = tf.train.Saver()
        if model_name is not None:
            saver.restore(sess, model_name)
        # Setup TensorBoard Writer
        writer = tf.summary.FileWriter("tensorboard/log")
        writer.add_graph(sess.graph)
        ## Losses
        tf.summary.scalar("Loss", self.loss)
        tf.summary.scalar("Accuracy", self.accuracy)
        write_op = tf.summary.merge_all()

        best_acc = 0
        for step in range(n_epoh):
            sum_loss = 0
            batches = batch(X, Y, batch_size=150)
            n_batches = 0
            for X_train, Y_train in batches:
                _, loss_ = sess.run([self.train_op, self.loss], {self.input_: X_train, self.tf_y: Y_train})
                sum_loss += loss_
                n_batches += 1
            avg_loss = sum_loss / n_batches
            if step % 1 == 0:
                accuracy_, summary = sess.run([self.accuracy, write_op], {self.input_: X_dev, self.tf_y: Y_dev})
                writer.add_summary(summary, step)
                print('Step:', step, '| train loss: %.4f' % avg_loss, '| test accuracy: %.2f' % accuracy_)
                if accuracy_ > best_acc:
                    best_acc = accuracy_
                    save_path = saver.save(sess, 'models/params', write_meta_graph=False)


def test(model_name):
    tf.reset_default_graph()
    cnn = CNN(IMAGES_SHAPE, N_LABELS)
    X_test, Y_test = load_test_dataset()
    sess = tf.Session()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    saver = tf.train.Saver()
    saver.restore(sess, model_name)
    accuracy_ = sess.run(cnn.accuracy, {cnn.input_: X_test, cnn.tf_y: Y_test})
    print("acc: " + str(accuracy_))


if __name__ == '__main__':
    # cnn = CNN(IMAGES_SHAPE, N_LABELS)
    # cnn.train()
    test("models/params")
