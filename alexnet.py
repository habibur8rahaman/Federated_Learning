# AlexNet Model Architecture
# AlexNet is complex model and will take more time to train. But it will also show better accuracy
# Load this instead of LeNet to use AlexNet for all clients


def build_model(input_shape=input_shape, num_classes=10):
    # with tf.device('/cpu:0'):
    with tf.device('/gpu:0'):  # Complex Model Architecture, Running on GPU will reduce training time significantly

        inputs = Input(shape=input_shape)
        # Conv1: 32 filters (original: 96), 5x5, stride=1
        x = Conv2D(32, (5, 5), strides=1, activation='relu', padding='same', name='Conv1')(inputs)
        # Conv2: 64 filters (original: 256), 3x3
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='Conv2')(x)
        x = MaxPooling2D(pool_size=(2, 2), name='Pool2')(x)
        # Conv3-5: 64 filters (original: 384/256)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='Conv3')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='Conv4')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='Conv5')(x)
        x = MaxPooling2D(pool_size=(2, 2), name='Pool3')(x)
        # Flatten
        x = Flatten(name='Flatten')(x)
        # Dense layers: 512 units (original: 4096)
        x = Dense(512, activation='relu', name='FC1')(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu', name='FC2')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(num_classes, activation='softmax', name='Output')(x)
    return Model(inputs, outputs, name='AlexNet_FashionMNIST_Light')
