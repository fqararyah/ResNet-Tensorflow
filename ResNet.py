import time
from ops import *
from utils import *
from tensorflow.python.client import timeline

from graphviz import Digraph
import json


def profile(run_metadata, epoch=0):
    with open('profs/timeline_step' + str(epoch) + '.json', 'w') as f:
        # Create the Timeline object, and write it to a json file
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        f.write(chrome_trace)


def graph_to_dot(graph):
    dot = Digraph()
    for n in graph.as_graph_def().node:
        dot.node(n.name, label=n.name)
        for i in n.input:
            dot.edge(i, n.name)
    return dot


class ResNet(object):
    def __init__(self, sess, args):
        self.model_name = 'ResNet'
        self.sess = sess
        self.dataset_name = args.dataset

        if self.dataset_name == 'cifar10':
            self.train_x, self.train_y, self.test_x, self.test_y = load_cifar10()
            self.img_size = 32
            self.c_dim = 3
            self.label_dim = 10

        if self.dataset_name == 'cifar100':
            self.train_x, self.train_y, self.test_x, self.test_y = load_cifar100()
            self.img_size = 32
            self.c_dim = 3
            self.label_dim = 100

        if self.dataset_name == 'mnist':
            self.train_x, self.train_y, self.test_x, self.test_y = load_mnist()
            self.img_size = 28
            self.c_dim = 1
            self.label_dim = 10

        if self.dataset_name == 'fashion-mnist':
            self.train_x, self.train_y, self.test_x, self.test_y = load_fashion()
            self.img_size = 28
            self.c_dim = 1
            self.label_dim = 10

        if self.dataset_name == 'tiny':
            self.train_x, self.train_y, self.test_x, self.test_y = load_tiny()
            self.img_size = 64
            self.c_dim = 3
            self.label_dim = 200

        self.checkpoint_dir = args.checkpoint_dir
        self.log_dir = args.log_dir

        self.res_n = args.res_n

        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.iteration = len(self.train_x) // self.batch_size

        self.init_lr = args.lr

    ##################################################################################
    # Generator
    ##################################################################################

    def network(self, x, is_training=True, reuse=False):
        with tf.variable_scope("network", reuse=reuse):

            if self.res_n < 50:
                residual_block = resblock
            else:
                residual_block = bottle_resblock

            residual_list = get_residual_layer(self.res_n)

            ch = 8  # paper is 64
            x = conv(x, channels=ch, kernel=3, stride=1, scope='conv')

            for i in range(residual_list[0]):
                x = residual_block(x, channels=ch, is_training=is_training,
                                   downsample=False, scope='resblock0_' + str(i))

            ########################################################################################################

            x = residual_block(
                x, channels=ch*2, is_training=is_training, downsample=True, scope='resblock1_0')

            for i in range(1, residual_list[1]):
                x = residual_block(x, channels=ch*2, is_training=is_training,
                                   downsample=False, scope='resblock1_' + str(i))

            ########################################################################################################

            x = residual_block(
                x, channels=ch*4, is_training=is_training, downsample=True, scope='resblock2_0')

            for i in range(1, residual_list[2]):
                x = residual_block(x, channels=ch*4, is_training=is_training,
                                   downsample=False, scope='resblock2_' + str(i))

            ########################################################################################################

            x = residual_block(
                x, channels=ch*8, is_training=is_training, downsample=True, scope='resblock_3_0')

            for i in range(1, residual_list[3]):
                x = residual_block(x, channels=ch*8, is_training=is_training,
                                   downsample=False, scope='resblock_3_' + str(i))

            ########################################################################################################

            x = batch_norm(x, is_training, scope='batch_norm')
            x = relu(x)

            x = global_avg_pooling(x)
            x = fully_conneted(x, units=self.label_dim, scope='logit')

            return x

    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        """ Graph Input """
        self.train_inptus = tf.placeholder(tf.float32, [
                                           self.batch_size, self.img_size, self.img_size, self.c_dim], name='train_inputs')
        self.train_labels = tf.placeholder(
            tf.float32, [self.batch_size, self.label_dim], name='train_labels')

        self.test_inptus = tf.placeholder(tf.float32, [len(
            self.test_x), self.img_size, self.img_size, self.c_dim], name='test_inputs')
        self.test_labels = tf.placeholder(
            tf.float32, [len(self.test_y), self.label_dim], name='test_labels')

        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        """ Model """
        self.train_logits = self.network(self.train_inptus)
        self.test_logits = self.network(
            self.test_inptus, is_training=False, reuse=True)

        self.train_loss, self.train_accuracy = classification_loss(
            logit=self.train_logits, label=self.train_labels)
        self.test_loss, self.test_accuracy = classification_loss(
            logit=self.test_logits, label=self.test_labels)

        reg_loss = tf.losses.get_regularization_loss()
        self.train_loss += reg_loss
        self.test_loss += reg_loss

        """ Training """
        self.optim = tf.train.MomentumOptimizer(
            self.lr, momentum=0.9).minimize(self.train_loss)

        """" Summary """
        self.summary_train_loss = tf.summary.scalar(
            "train_loss", self.train_loss)
        self.summary_train_accuracy = tf.summary.scalar(
            "train_accuracy", self.train_accuracy)

        self.summary_test_loss = tf.summary.scalar("test_loss", self.test_loss)
        self.summary_test_accuracy = tf.summary.scalar(
            "test_accuracy", self.test_accuracy)

        self.train_summary = tf.summary.merge(
            [self.summary_train_loss, self.summary_train_accuracy])
        self.test_summary = tf.summary.merge(
            [self.summary_test_loss, self.summary_test_accuracy])

    ##################################################################################
    # Train
    ##################################################################################

    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(
            self.log_dir + '/' + self.model_dir, self.sess.graph)

        # fareed
        dot_rep = graph_to_dot(self.sess.graph)
        with open('./resnet.dot', 'w') as fwr:
            fwr.write(str(dot_rep))

        options = tf.RunOptions(
            trace_level=tf.RunOptions.SOFTWARE_TRACE)
        run_metadata = tf.RunMetadata()

        operations_tensors = {}
        operations_names = self.sess.graph.get_operations()
        count1 = 0
        count2 = 0

        for operation in operations_names:
            operation_name = operation.name
            operations_info = self.sess.graph.get_operation_by_name(
                operation_name).values()
            if len(operations_info) > 0:
                if not (operations_info[0].shape.ndims is None):
                    operation_shape = operations_info[0].shape.as_list()
                    operation_dtype_size = operations_info[0].dtype.size
                    if not (operation_dtype_size is None):
                        operation_no_of_elements = 1
                        for dim in operation_shape:
                            if not(dim is None):
                                operation_no_of_elements = operation_no_of_elements * dim
                        total_size = operation_no_of_elements * operation_dtype_size
                        operations_tensors[operation_name] = total_size
                    else:
                        count1 = count1 + 1
                else:
                    count1 = count1 + 1
                    operations_tensors[operation_name] = -1

                #   print('no shape_1: ' + operation_name)
                #  print('no shape_2: ' + str(operations_info))
                #  operation_namee = operation_name + ':0'
                # tensor = tf.get_default_graph().get_tensor_by_name(operation_namee)
                # print('no shape_3:' + str(tf.shape(tensor)))
                # print('no shape:' + str(tensor.get_shape()))

            else:
                # print('no info :' + operation_name)
                # operation_namee = operation.name + ':0'
                count2 = count2 + 1
                operations_tensors[operation_name] = -1

                # try:
                #   tensor = tf.get_default_graph().get_tensor_by_name(operation_namee)
                # print(tensor)
                # print(tf.shape(tensor))
                # except:
                # print('no tensor: ' + operation_namee)
                print(count1)
                print(count2)

                with open('./tensors_sz.json', 'w') as f:
                    json.dump(operations_tensors, f)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            epoch_lr = self.init_lr
            start_epoch = (int)(checkpoint_counter / self.iteration)
            start_batch_id = checkpoint_counter - start_epoch * self.iteration
            counter = checkpoint_counter

            if start_epoch >= int(self.epoch * 0.75):
                epoch_lr = epoch_lr * 0.01
            elif start_epoch >= int(self.epoch * 0.5) and start_epoch < int(self.epoch * 0.75):
                epoch_lr = epoch_lr * 0.1
            print(" [*] Load SUCCESS")
        else:
            epoch_lr = self.init_lr
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):
            if epoch == int(self.epoch * 0.5) or epoch == int(self.epoch * 0.75):
                epoch_lr = epoch_lr * 0.1

            # get batch data
            for idx in range(start_batch_id, self.iteration):
                batch_x = self.train_x[idx *
                                       self.batch_size:(idx+1)*self.batch_size]
                batch_y = self.train_y[idx *
                                       self.batch_size:(idx+1)*self.batch_size]

                batch_x = data_augmentation(
                    batch_x, self.img_size, self.dataset_name)

                train_feed_dict = {
                    self.train_inptus: batch_x,
                    self.train_labels: batch_y,
                    self.lr: epoch_lr
                }

                test_feed_dict = {
                    self.test_inptus: self.test_x,
                    self.test_labels: self.test_y
                }

                # update network
                if idx % 5 == 0:
                    _, summary_str, train_loss, train_accuracy = self.sess.run(
                        [self.optim, self.train_summary, self.train_loss, self.train_accuracy], feed_dict=train_feed_dict, run_metadata=run_metadata, options=options)
                    profile(run_metadata, str(epoch) + '_' + str(idx))

                else:
                    st_time = time.time()
                    _, summary_str, train_loss, train_accuracy = self.sess.run(
                        [self.optim, self.train_summary, self.train_loss, self.train_accuracy], feed_dict=train_feed_dict)
                    print('step_time' + str(time.time() - st_time))

                self.writer.add_summary(summary_str, counter)

                # test
                summary_str, test_loss, test_accuracy = self.sess.run(
                    [self.test_summary, self.test_loss, self.test_accuracy], feed_dict=test_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%5d/%5d] time: %4.4f, train_accuracy: %.2f, test_accuracy: %.2f, learning_rate : %.4f"
                      % (epoch, idx, self.iteration, time.time() - start_time, train_accuracy, test_accuracy, epoch_lr))

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            self.save(self.checkpoint_dir, counter)

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    @property
    def model_dir(self):
        return "{}{}_{}_{}_{}".format(self.model_name, self.res_n, self.dataset_name, self.batch_size, self.init_lr)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(
            checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(
                checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def test(self):
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)

        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        test_feed_dict = {
            self.test_inptus: self.test_x,
            self.test_labels: self.test_y
        }

        test_accuracy = self.sess.run(
            self.test_accuracy, feed_dict=test_feed_dict)
        print("test_accuracy: {}".format(test_accuracy))
