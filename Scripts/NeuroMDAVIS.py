import numpy as np
import tensorflow as tf
import keras
import keras.optimizers
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers

# NeuroMDAVIS architecture
def NeuroMDAVIS(X, dim, lambda_act, lambda_weight, num_neuron, bs, epoch, sd, verbose):
    num_in_neuron = X[0].shape[0]
    data = np.eye(num_in_neuron)
    num_out_neuron = [i.shape[1] for i in X] 
    n = len(X)
    
    name = ["modality"+str(i+1)+"_output" for i in range(n)]
    
    inputs = tf.keras.Input(shape=(num_in_neuron,))

    layer1 = keras.layers.Dense(dim, activation="linear",
                      activity_regularizer=regularizers.l2(lambda_act),
                      kernel_regularizer=regularizers.l2(lambda_weight), 
                      kernel_initializer=tf.keras.initializers.RandomNormal(seed=sd),
                      name = 'Latent_Layer',
                      )(inputs)
    
    layer2 = Dense(num_neuron[0], activation="relu",
                      activity_regularizer=regularizers.l2(lambda_act),
                      kernel_regularizer=regularizers.l2(lambda_weight), 
                      kernel_initializer=tf.keras.initializers.he_uniform(seed=sd), 
                      bias_initializer = tf.keras.initializers.Constant(0.1),
                      name = 'Hidden_Layer_1',
                      )(layer1)
    
    
    layer3 = [Dense(num_neuron[1][i], activation="relu",
                      activity_regularizer=regularizers.l2(lambda_act),
                      kernel_regularizer=regularizers.l2(lambda_weight), 
                      kernel_initializer=tf.keras.initializers.he_uniform(seed=sd), 
                      bias_initializer = tf.keras.initializers.Constant(0.1),
                      )(layer2) for i in range(n)]
    
    outputs = [Dense(num_out_neuron[i], activation="linear",
                      kernel_initializer=tf.keras.initializers.GlorotUniform(seed=sd),
                      name=name[i],
                      )(layer3[i]) for i in range(n)]
    
    Neuromdavis = keras.Model(inputs=inputs, outputs=outputs)
    Low = keras.Model(inputs=inputs, outputs=layer1)
    
    
    losses = {name[i]:"mean_squared_error" for i in range(n)}
    
    lossWeights = {name[i]:1.0 for i in range(n)}

    
    Neuromdavis.compile(
        loss=losses,
        loss_weights=lossWeights,
        optimizer=tf.keras.optimizers.Adam()
    )
    
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
    Neuromdavis.fit(data, X, batch_size=bs, epochs=epoch, verbose=verbose, callbacks=[callback]) 
    low = Low.predict(data)

    return low

