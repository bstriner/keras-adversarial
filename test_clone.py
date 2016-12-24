
import keras.backend as K
import numpy as np

if K.backend() == "tensorflow":
    import tensorflow as tf


    def f_replace(f, replace):
        replacements = {k.op.outputs[0]: v.op.outputs[0] for k,v in replace.iteritems()}
        f2 = tf.contrib.graph_editor.graph_replace(f, replacements)
        return f2
else:
    import theano


    def f_replace(f, replace):
        return theano.clone(f, replace=replace)

x1 = K.variable(np.float32(2), name="x1")
x2 = K.variable(np.float32(3), name="x2")
y1 = K.pow(x1, 2)
f1 = K.function([], [y1])
print "F1: {}".format(f1([]))
y2 = f_replace(y1, {x1:x2})
f2 = K.function([], [y2])
print "F2: {}".format(f2([]))
