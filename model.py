import tensorflow
from keras.models import Model
from keras.layers import Input, Dense, Activation, Dropout,BatchNormalization

def nn_model(input_shape):
    xi=Input(shape=(input_shape,))

    x=Dense(256)(xi)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)
    x=Dropout(0.5)(x)

    x=Dense(64)(x)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)
    x=Dropout(0.5)(x)
    
    x=Dense(1)(x)
    
    model=Model(inputs=xi, outputs=x)
    return model