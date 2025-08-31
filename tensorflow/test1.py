import tensorflow as tf 
import numpy as np

## creating a tensor
string = tf.Variable("this is a string", tf.string)
number = tf.Variable(123, tf.int16)
floating = tf.Variable(3.14, tf.float16)

rank1_tensor = tf.Variable(["test", "tensor", "flow", "easy"], tf.string)
rank2_tensor = tf.Variable([["test", "tensor", "flow"], ["pretty", "easy", "itis"]], tf.string)

print(tf.rank(rank1_tensor))
print(tf.rank(rank2_tensor))

print(rank1_tensor.shape) # shape print it as number of rows and columns type thing
print(rank2_tensor.shape)

tensor_1 = tf.ones([1, 2, 3])
print(tensor_1)
tensor_1 = tf.ones([6, 4, 3]) # THIS WILL CRAETE A TENSOR OF 6 BLOCKS , 4 ROWS AND 3 COLUMNS OF ALL ONES
print(tensor_1)
# LETS TRY TO RESHAPE IT
reshaped_tensor = tf.reshape(tensor_1, [2, 4, 9])
print(reshaped_tensor)




