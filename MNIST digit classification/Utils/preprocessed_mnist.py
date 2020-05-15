import keras


def laod_dataset(flatten=False):
    # load the mnist dataset that comes with keras integration
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    # now as we have our dataset, you can check manually its shape and size
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    # (60000, 28, 28) (10000, 28, 28) (60000,) (10000,)

    # Let's normalize our image samples in the range 0 to 255
    X_train = X_train.astype(float)/255.0
    X_test = X_test.astype(float)/255.0

    # as validation is required in almost every project,
    # keep it as a good practice and reserve our validation set
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # last 10000 samples from our training set is reserved for validation

    # flatten our images into a vector
    if flatten:
        X_train = X_train.reshape([X_train.shape[0], -1])
        X_val = X_val.reshape([X_val.shape[0], -1])
        X_test = X_test.reshape([X_test.shape[0], -1])

    return X_train, y_train, X_val, y_val, X_test, y_test
    pass
