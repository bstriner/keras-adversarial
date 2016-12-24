
import keras.backend as K
import numpy as np

if K.backend() == "tensorflow":
    import tensorflow as tf


    def f_replace(f, replace):
        return tf.contrib.graph_editor.copy_with_input_replacements(f, replace)
else:
    import theano


    def f_replace(f, replace):
        return theano.clone(f, replace=replace)



x1 = K.variable(np.float32(2))
y1 = K.pow(x1, 2)
f1 = K.function([], y1)
x2 = K.variable(np.float32(3))
y2 = f_replace(y1, {x1:x2})
f2 = K.function([], y2)

print "F1: {}, F2:{}".format(f1(), f2())