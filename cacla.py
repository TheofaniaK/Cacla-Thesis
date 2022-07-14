import numpy as np
from tensorflow import keras
import keras
from keras.models import Sequential
from keras.layers import Dense

class Cacla:
    def __init__(self, input_d, output_d, a, b, g, lr_dec, exploration_dec, exploration_fac):
        """
        initializes CACLA reinforcement learning algorithm.
        #self.env = env
        """
        self.input_dim = input_d
        self.output_dim = output_d
        self.gamma = g
        self.exploration_factor = exploration_fac
        self.lr_decay = lr_dec
        self.exploration_decay = exploration_dec

        self.alpha = a
        self.beta = b

        # creates neural networks.
        self.actor = self._create_actor(input_d, output_d, a)
        self.critic = self._create_critic(input_d, 1, b)

    def update_lr(self, lr_dec):
        """
        :param lr_dec: decay for both actor and critic
        changes learning rate for actor and critic based on lr_decay.
        """
        keras.backend.set_value(self.critic.optimizer.lr,
                                keras.backend.get_value(self.critic.optimizer.lr) * lr_dec)
        keras.backend.set_value(self.actor.optimizer.lr,
                                keras.backend.get_value(self.actor.optimizer.lr) * lr_dec)
        self.alpha *= lr_dec
        self.beta *= lr_dec

    def update_exploration(self, exploration_dec=None):
        """
        updates the exploration factor.
        :param exploration_dec: exploration_factor multiplier. if None, default value is used.
        """
        if exploration_dec is None:
            exploration_dec = self.exploration_decay
        self.exploration_factor *= exploration_dec

    @staticmethod
    def sample(action, explore):
        """
        :param action: default action predicted by actor
        :param explore: exploration factor
        :return: explored action, normally distributed around default action.
        """
        a = [i + np.random.normal(0, 1) * explore for i in action]
        return a

    @staticmethod
    def _create_actor(input_d, output_d, learning_rate):
        """
        Creates actor. Uses 1 hidden layers with number of neurons 5 * input_dim (40).
        initializes weights to some small value.
        """
        l1_size = 20 * input_d

        model = Sequential()
        model.add(Dense(l1_size, input_dim=input_d, activation="relu",
                        kernel_initializer=keras.initializers.random_normal(0.0, np.sqrt(2 / input_d))))
        model.add(Dense(output_d, activation="linear",
                        kernel_initializer=keras.initializers.random_normal(0.0, np.sqrt(2 / l1_size))))

        adam = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(loss='mean_squared_error', optimizer=adam)
        return model

    @staticmethod
    def _create_critic(input_d, output_d, learning_rate):
        """
        See self._create_actor.
        """
        l1_size = 20 * input_d

        model = Sequential()
        model.add(Dense(l1_size, input_dim=input_d, activation="relu",
                        kernel_initializer=keras.initializers.random_normal(0.0, np.sqrt(2 / input_d))))
        model.add(Dense(output_d, activation='linear',
                        kernel_initializer=keras.initializers.random_normal(0.0, np.sqrt(2 / l1_size))))

        adam = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(loss='mean_squared_error', optimizer=adam)
        return model
