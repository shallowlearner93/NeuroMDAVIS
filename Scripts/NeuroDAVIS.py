import numpy as np
import tensorflow as tf
import keras
import keras.optimizers
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers


# neuroDAVIS architecture
def NeuroDAVIS(X, dim, lambda_act, lambda_weight, num_neuron, bs, epoch, sd, verbose):
    num_in_neuron = X.shape[0]
    data = np.eye(num_in_neuron)
    num_out_neuron = X.shape[1]
    
    inputs = tf.keras.Input(shape=(num_in_neuron,))

    layer1 = keras.layers.Dense(dim, activation="linear",
                      activity_regularizer=regularizers.l2(lambda_act),
                      kernel_regularizer=regularizers.l2(lambda_weight), 
                      kernel_initializer=tf.keras.initializers.RandomNormal(seed=sd),
                      name = 'Latent_Layer',
                      )(inputs)
    
    layer2 = keras.layers.Dense(num_neuron[0], activation="relu",
                      activity_regularizer=regularizers.l2(lambda_act),
                      kernel_regularizer=regularizers.l2(lambda_weight),
                      kernel_initializer=tf.keras.initializers.GlorotUniform(seed=sd),
                      bias_initializer = tf.keras.initializers.Constant(0.1),
                      name = 'Hidden_Layer_1',
                      )(layer1)
       
    layer3 = keras.layers.Dense(num_neuron[1], activation="relu",
                      activity_regularizer=regularizers.l2(lambda_act),
                      kernel_regularizer=regularizers.l2(lambda_weight),
                      kernel_initializer=tf.keras.initializers.GlorotUniform(seed=sd),
                      bias_initializer = tf.keras.initializers.Constant(0.1),
                      name = 'Hidden_Layer_2',          
                      )(layer2)

    outputs = keras.layers.Dense(num_out_neuron, activation="linear",
                      kernel_initializer=tf.keras.initializers.GlorotUniform(seed=sd),
                      bias_initializer = tf.keras.initializers.Zeros(),
                      name = 'Reconstruct_Layer',
                      )(layer3)

    neuroDAVIS = keras.Model(inputs=inputs, outputs=outputs)
    Low = keras.Model(inputs=inputs, outputs=layer1)

    neuroDAVIS.compile(
        loss=keras.losses.mean_squared_error,
        optimizer=tf.keras.optimizers.Adam()
    )
    
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
    neuroDAVIS.fit(data, X, batch_size=bs, epochs=epoch, verbose=verbose, callbacks=[callback])  
    
    low = Low.predict(data)

    return low