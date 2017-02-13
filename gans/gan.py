'''
An example of a GAN used to learn a 1D random Norman Gaussian distribution for exploration / hacking on GANs

This example was heavily influenced by https://github.com/AYLIEN/gan-intro/blob/master/gan.py, while also working
through the original GAN paper by Ian Goodfellow et. al.: https://arxiv.org/abs/1406.2661, as well.

Hoping to carry it into something more robust as I learn and explore.
'''

from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

# TODO : Rebuild Docker image with seaborn and animate graphs

import argparse
import numpy as np
from scipy.stats import norm
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import animation

seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)

class DataDistribution(object):
    '''
    Generate some data
    '''

    def __init__(self):
        self.mu = 4
        self.sigma = 0.5

    def sample(self, N):
        '''
        Generate N samples of data and quicksort them
        :param N: number of samples
        :return: sorted array of sample data
        '''

        samples = np.random.normal(self.mu, self.sigma, N)
        # quicksort is numpy's default sort algorithm
        samples.sort()
        return samples

class GeneratorDistribution(object):
    '''
    Generate initial distribution for generator network
    '''

    def __init__(self, range):
        self.range = range

    def sample(self, N):
        '''
        Generate N perturbed samples of data in (-range, range) interval for use with generator network

        source : http://blog.evjang.com/2016/06/generative-adversarial-nets-in.html

        :param N: number of samples
        :return: array of samples
        '''

        return np.linspace(-self.range, self.range, N) + np.random.random(N) * 0.01


# TODO : Investigate why this linearity was introduced
def linear(input, output_dimension, scope=None, stddev=1.0):
    '''
    Perform linear transformation over input

    :param input:
    :param output_dimension:
    :param scope:
    :param stddev:
    :return:
    '''

    norm = tf.random_normal(stddev=stddev)
    const = tf.constant_initializer(0.0)
    with tf.variable_scope(scope or 'linear'): 
        w = tf.get_variable('w', [input.get_shape()[1], output_dimension], initializer=norm)
        b = tf.get_variable('b', [output_dimension], initializer=const)
        return tf.matmul(input, w) + b

def generator_network(input, hidden_dimension):
    '''
    Generator architecture

    :param input: generator distribution samples
    :param hidden_dimensions: hidden dimensions
    :return: hidden_1
    '''

    hidden_0 = tf.nn.softplus(linear(input, hidden_dimension, 'generator_0'))
    hidden_1 = linear(hidden_0, 1, 'generator_1')

    return hidden_1

# TODO : Implement minibatch according to https://arxiv.org/pdf/1606.03498.pdf
def discriminator_network(input, hidden_dimension):
    '''
    Discriminator architecture

    :param input: generator or data distribution samples
    :param hidden_dimension:
    :return:
    '''

    hidden_0 = tf.tanh(linear(input, hidden_dimension * 2, scope='discriminator_0'))
    hidden_1 = tf.tanh(linear(hidden_0, hidden_dimension * 2, scope='discriminator_1'))
    hidden_2 = tf.tanh(linear(hidden_1, hidden_dimension * 2, scope='discriminator_2'))

    return hidden_2

def optimizer(loss, variable_list, initial_learning_rate):
    '''
    Optimizer of model
    :param loss: Loss function
    :param variable_list:
    :param initial_learning_rate: Initial learning rate
    :return: optimizer
    '''

    decay = 0.95
    num_decay_steps = 150
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                               batch,
                                               num_decay_steps,
                                               decay,
                                               staircase=True)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,
                                                                          global_step=batch,
                                                                          var_list=variable_list)

    return optimizer

class GAN(object):
    """Generative Adversarial Network
        :param data: input data
        :param gen:
        :param num_steps: number of epochs
        :param batch_size: data batch_size
        :param log_every: log rate
        :param mlp_hidden_size: side of MLP used in generator and discriminator
        :param anim_path: animation path for output
        :param anim_frames: array for graphs for animation frames
        """

    def __init__(self, data, gen, num_steps, batch_size, log_every, anim_path):


        self.data = data
        self.gen = gen
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.log_every = log_every
        self.mlp_hidden_size = 4
        self.anim_path = anim_path
        self.anim_frames = []

        self.learning_rate = 0.03

        self._create_model()

    def _create_model(self):

        # Pretrain the discriminator over a sample set of data
        # TODO: Understand why
        with tf.variable_scope('discriminator_pretrain'):
            self.pre_input = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
            self.pre_labels = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
            discriminator_pretrain = discriminator_network(self.pre_input, self.mlp_hidden_size)
            self.pre_loss = tf.reduce_mean(tf.square(discriminator_pretrain - self.pre_labels))
            self.pre_optimizer = optimizer(self.pre_loss, None, self.learning_rate)

        # define generator network
        with tf.variable_scope('generator'):
            self.z = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
            self.generator = generator_network(self.z, self.mlp_hidden_size)

        # create two copies of the discriminator as TensorFlow doesn't support the same network
        # with different inputs
        with tf.variable_scope('discriminator') as scope:
            self.x = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
            self.discriminator_0 = discriminator_network(self.x, self.mlp_hidden_size)
            scope.reuse_variables()
            self.discriminator_1 = discriminator_network(self.generator, self.mlp_hidden_size)

        # define losses for both generator and discriminator
        # note the comparison of the discriminator loss of the two networks
        self.loss_discriminator = tf.reduce_mean(-tf.log(self.discriminator_0) - tf.log(1 - self.discriminator_1))
        self.loss_generator = tf.reduce_mean(-tf.log(self.discriminator_1))

        self.discriminator_pretrain_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator_pretrain')
        self.discriminator_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        self.generator_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

        # optimizers
        self.discriminator_optimizer = optimizer(self.loss_discriminator, self.discriminator_params, self.learning_rate)
        self.generator_optimizer = optimizer(self.loss_generator, self.generator_params, self.learning_rate)


    def train(self):

        with tf.Session() as session:
            tf.initialize_all_variables()

            # pretrain the discriminator
            num_pretrain_steps = 1000
            for step in xrange(num_pretrain_steps):
                d = (np.random.random(self.batch_size) - 0.5) * 10.0
                labels = norm.pdf(d, loc=self.data.mu, scale=self.data.sigma)
                pretrain_loss, _ = session.run([self.pre_loss, self.pre_optimizer], {
                    self.pre_input: np.reshape(d, (self.batch_size, 1)),
                    self.pre_labels: np.reshape(labels, (self.batch_size, 1))
                })

            self.discriminator_weights = session.run(self.discriminator_pretrain_params)


            # copy the weights from pre-training over to new D network
            for i, v in enumerate(self.discriminator_params):
                session.run(v.assign(self.discriminator_weights[i]))

            for step in xrange(self.num_steps):
                # update discriminator
                x = self.data.sample(self.batch_size)
                z = self.gen.sample(self.batch_size)
                loss_discriminator  = session.run([self.loss_discriminator, self.discriminator_optimizer], {
                    self.x: np.reshape(x, (self.batch_size, 1)),
                    self.z: np.reshape(z, (self.batch_size, 1))
                })

                # update generator
                z = self.gen.sample(self.batch_size)
                loss_generator = session.run([self.loss_generator, self.generator_optimizer], {
                    self.z: np.reshape(z, (self.batch_size, 1))
                })

                if step % self.log_every == 0:
                    print('{}: {}\t{}'.format(step, loss_discriminator, loss_generator))

                if self.anim_path:
                    self.anim_frames.append(self._samples(session))

            if self.anim_path:
                self._save_animation()
            else:
                self._plot_distributions(session)


    def _samples(self, session, num_points=10000, num_bins=100):
        """
        Return a tuple (bd, pd, pg) where db is the current decision boundary,
        pd is a histogram of samples from the data distribution, and pg is a
        histogram of generated samples

        :param session: TensorFlow session
        :param num_points: number of points to sample
        :param num_bins: number of bins of samples
        :return: (db, pd, pg)
        """

        xs = np.linspace(-self.gen.range, self.gen.range, num_points)
        bins = np.linspace(-self.gen.range, self.gen.range, num_bins)

        # decision boundary
        db = np.zeros((num_points, 1))
        for i in range(num_points // self.batch_size):
            db[self.batch_size * i:self.batch_size * (i + 1)] = session.run(self.discriminator_1, {
                self.x: np.reshape(
                    xs[self.batch_size * i:self.batch_size * (i + 1)],
                    (self.batch_size,1)
                )
            })


        # data distribution
        d = self.data.sample(num_points)
        pd, _ = np.histogram(d, bins=bins, density=True)


        # generated samples
        zs = np.linspace(-self.gen.range, self.gen.range, num_points)
        g = np.zeros((num_points, 1))

        for i in range(num_points // self.batch_size):
            g[self.batch_size* i:self.batch_size * (i + 1)] = session.run(self.generator, {
                self.z: np.reshape(
                    zs[self.batch_size * i:self.batch_size * (i + 1)],
                    (self.batch_size, 1)
                )
            })

        pg, _ = np.histogram(g, bins=bins, density=True)


        return db, pd, pg



    def _plot_distributions(self, session):
        db, pd, pg = self._samples(session)
        db_x = np.linspace(-self.gen.range, self.gen.range, len(db))
        p_x = np.linspace(-self.gen.range, self.gen.ragne, len(pd))
        f, ax = plt.subplots(1)
        ax.plot(db_x, db, label='decision boundary')
        ax.set_ylim(0, 1)
        plt.plot(p_x, pd, label='real data')
        plt.plot(p_x, pg, label='generated data')
        plt.title=('1D Generative Adversarial Network')
        plt.xlabel('Data Values')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.show()



    def _save_animation(self):
        f, ax = plt.subplots(figsize=(6, 4))
        f.suptitle('1D Generative Adversarial Network', fontsize=15)
        plt.xlabel('Data values')
        plt.ylabel('Probability Density')
        ax.set_xlim(-6, 6)
        ax.set_ylim(0, 1.4)
        line_db, = ax.plot([], [], label='decision boundary')
        line_pd, = ax.plot([], [], label='real data')
        line_pg, = ax.plot([], [], label='generated data')
        frame_number = ax.text(
            0.02,
            0.95,
            '',
            horizonalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes
        )
        ax.legend()

        db, pd, _ = self.anim_frames[0]
        db_x = np.linspace(-self.gen.range, self.gen.range, len(db))
        p_x = np.linspace(-self.gen.range, self.gen.range, len(pd))

        def init():
            line_db.set_data([], [])
            line_pd.set_data([], [])
            line_pg.set_data([], [])
            frame_number.set_text('')
            return(line_db, line_pd, line_pg, frame_number)


        def animate(i):
            frame_number.set_text(
                'Frame: {}/{}'.format(i, len(self.anim_frames))
            )

            db, pd, pg = self.anim_frames[i]
            line_db.set_data(db_x, db)
            line_pd.set_data(p_x, pd)
            line_pg.set_data(p_x, pg)
            return (line_db, line_pd, line_pg, frame_number)

        anim = animation.FuncAnimation(
            f,
            animate,
            init_func=init,
            frames=len(self.anim_frames),
            blit=True
        )
        anim.save(self.anim_path, fps=30, extra_args=['-vcodec', 'libx264'])

def main(args):
    model = GAN(
        DataDistribution(),
        GeneratorDistribution(range=8),
        args.num_steps,
        args.batch_size,
        args.log_every,
        args.anim
    )

    model.train()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-steps', type=int, default=1200,
                        help='the number of training steps to take')
    parser.add_argument('--batch-size', type=int, default=12,
                        help='the batch size')
    parser.add_argument('--log-every', type=int, default=10,
                        help='print loss after this many steps')
    parser.add_argument('--anim', type=str, default=None,
                        help='name of the output animation file (default: none)')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())