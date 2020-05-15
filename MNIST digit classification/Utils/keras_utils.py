# @author Mrutyunjay Biswal
# I am using tf < 2

import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import keras


# !!! remember to clear session/graph if you rebuild your graph to avoid out-of-memory errors !!!

def reset_tf_session():
    curr_session = tf.get_default_session()
    # close current session
    if curr_session is not None:
        curr_session.close()
    # reset graph
    keras.backend.clear_session()
    # create new session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    s = tf.InteractiveSession(config=config)
    keras.backend.set_session(s)
    return s
