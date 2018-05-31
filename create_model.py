import keras
from keras.models import Sequential
from keras.layers import \
    Activation, \
    BatchNormalization, \
    Convolution2D, \
    Dense, \
    Dropout, \
    Embedding, \
    Flatten, \
    GlobalAveragePooling2D, \
    LSTM, \
    MaxPooling2D, \
    Conv2D

from keras.optimizers import SGD

def create_model(input_shape, output_size, loss_function, dataset):
    if dataset == 'cifar100':
        num_classes = 100
    else:
        num_classes = 10
    print ("Dataset", dataset)
    print (" output size", output_size)

    if dataset == 'mnist' or dataset == 'fmnist' or dataset == "im-mnist":
        print ("--------------", input_shape)
        model = Sequential ([
            # conv1_*
            Convolution2D (32, kernel_size=3, padding="same",
                           input_shape=input_shape),
            Activation ("relu"),
            Convolution2D (32, kernel_size=3, padding="same"),
            Activation ("relu"),
            MaxPooling2D (pool_size=(2, 2)),
            Dropout (0.25),

            # conv2_*
            Convolution2D (64, kernel_size=3, padding="same"),
            Activation ("relu"),
            Convolution2D (64, kernel_size=3, padding="same"),
            Activation ("relu"),
            MaxPooling2D (pool_size=(2, 2)),
            Dropout (0.25),

            # Fully connected
            Flatten (),
            Dense (512),
            Activation ("relu"),
            Dropout (0.5),
            Dense (512, name="features"),
            Activation ("relu"),
            Dropout (0.5),
            Dense (output_size),
            Activation ("softmax", name="prob")
        ])

        model.compile (
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"]
        )

        return model

    elif dataset == 'cifar10':
        kwargs = {
            "activation": "relu",
            "border_mode": "same"
        }
        model = Sequential ([
        Conv2D (32, (3, 3), padding='same',
                           input_shape=input_shape),
        Activation ('relu'),
        Conv2D (32, (3, 3)),
        Activation ('relu'),
        MaxPooling2D (pool_size=(2, 2)),
        Dropout (0.25),

        Conv2D (64, (3, 3), padding='same'),
        Activation ('relu'),
        Conv2D (64, (3, 3)),
        Activation ('relu'),
        MaxPooling2D (pool_size=(2, 2)),
        Dropout (0.25),

        Flatten (),
        Dense (512, name="features"),
        Activation ('relu'),
        Dropout (0.5),
        Dense (output_size),
        Activation ('softmax', name="prob")
        ])

        model.compile (
            loss="categorical_crossentropy",
            optimizer=SGD (momentum=0.9),
            metrics=["accuracy"]
        )

        return model

    elif dataset == 'cifar100':
        model = keras.applications.resnet50.ResNet50 (include_top=False,
                                                      weights=None,
                                                      input_tensor=None,
                                                      input_shape=input_shape,
                                                      pooling='max',
                                                      classes=100)
        model.add (Flatten ())
        model.add (Dense (512))
        model.add (Activation ('relu'))
        model.add (Dropout (0.5))
        model.add (Dense (num_classes))
        model.add (Activation ('softmax'))
        return model

    elif dataset == 'ptb':
        vocab_size = 10000
        output_size = vocab_size
        print ("input_shape", input_shape)
        model = Sequential ([
            Embedding (vocab_size + 1, 64, mask_zero=True,
                       input_length=input_shape[0], name='emb1'),
            LSTM (256, unroll=False, return_sequences=True, name='lstm1'),
            Dropout (0.5),
            LSTM (256, unroll=False, name='lstm2'),
            Dropout (0.5),
            Dense (output_size, name='dense1'),
            Activation ("softmax")
        ])

        model.compile (
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        return model

    elif dataset == 'svnh':
        model = Sequential ()

        model.add (Conv2D (64, (5, 5), padding='same',
                           input_shape=input_shape))
        model.add (Activation ('relu'))
        model.add (Conv2D (64, (5, 5)))
        model.add (Activation ('relu'))
        model.add (MaxPooling2D (pool_size=(2, 2)))
        model.add (Dropout (0.25))

        model.add (Conv2D (32, (5, 5), padding='same'))
        model.add (Activation ('relu'))
        model.add (MaxPooling2D (pool_size=(2, 2)))
        model.add (Dropout (0.25))

        model.add (Flatten ())
        model.add (Dense (1024, name="features"))
        model.add (Activation ('relu'))
        model.add (Dropout (0.5))
        model.add (Dense (num_classes))
        model.add (Activation ('softmax', name="prob"))

        # initiate RMSprop optimizer
        opt = keras.optimizers.RMSprop ()

        # Let's train the model using RMSprop
        model.compile (loss=loss_function,
                       optimizer=opt,
                       metrics=['accuracy'])

        return model