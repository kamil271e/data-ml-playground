from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, AveragePooling2D, Flatten, Dense
from keras.models import Model

def conv_block(input_tensor, num_filters):
    x = Conv2D(num_filters, kernel_size=3, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)

    shortcut = Conv2D(num_filters, kernel_size=1, padding='same')(input_tensor) # 1x1 conv
    shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    return x

def resnet(input_shape, num_classes):
    input_tensor = Input(shape=input_shape)
    x = Conv2D(64, kernel_size=7, strides=2, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = conv_block(x, 64)
    x = conv_block(x, 64)
    x = conv_block(x, 128)
    x = conv_block(x, 128)
    x = conv_block(x, 256)
    x = conv_block(x, 256)
    x = conv_block(x, 512)
    x = conv_block(x, 512)

    x = AveragePooling2D(pool_size=7)(x)
    x = Flatten()(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=x)

    return model


m = resnet((28,28,3), 10)
