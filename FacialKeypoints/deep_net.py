import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

TRAIN_STEPS = 1600
BATCH_SIZE = 100
DROPOUT = 1.0
VALIDATION_PERCENT = 0.1
SEED = 14
EARLY_STOPPING = 25

def convert_data(data):
    cols = data.columns[:-1]
    data['Image'] = data['Image'].apply(lambda x: np.fromstring(x, sep=' ') / 255.0)
    data = data.dropna()
    for col in cols:
        data[col] = data[col].apply(lambda x: x/96.0)
    return data
    
def get_sample(data, batch_size):
    s = data.sample(n=batch_size)
    x = s.pop('Image')
    x = np.vstack(x)
    y_ = s.as_matrix()
    return x, y_
    
def plot_img(x, y, y_):
    img = np.reshape(x, [96,96])
    y = y[0]*96
    y_ = y_[0]*96.0
    plt.imshow(img, cmap='gray')
    if y is not None:
        plt.scatter(y[0::2], y[1::2], c='r', marker='x')
    if y_ is not None:
        plt.scatter(y_[0::2], y_[1::2], c='g', marker='o')
    
    plt.show()

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def conv_2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')

def max_pool_nxn(x, stride):
    return tf.nn.max_pool(x, ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1], padding='SAME')

def create_network():
    #Define the input and reshape it.
    input_layer = tf.placeholder('float', [None, 9216])
    x = tf.reshape(input_layer, [-1, 96, 96, 1])

    #Define weights, biases.
    W_conv1 = weight_variable([8, 8, 1, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    #3969 = 7*7*81, image is 7x7 after pooling twice
    W_fcl1 = weight_variable([9216, 256])
    b_fcl1 = bias_variable([256])

    W_fcl2 = weight_variable([256, 30])
    b_fcl2 = bias_variable([30])

    #Hidden layers.
    h_conv1 = tf.nn.relu(conv_2d(x, W_conv1, 1) + b_conv1)
    h_pool1 = max_pool_nxn(h_conv1, 4)

    h_conv2 = tf.nn.relu(conv_2d(h_pool1, W_conv2, 1) + b_conv2)
    h_pool2 = max_pool_nxn(h_conv2, 2)

    h_flat = tf.reshape(h_pool2, [-1, 9216])
    
    h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fcl1) + b_fcl1)
 
    h_fc1 = tf.nn.dropout(h_fc1, keep_prob=DROPOUT, seed=SEED)
    
    readout = tf.matmul(h_fc1, W_fcl2) + b_fcl2

    return input_layer, readout

def train_network(x, y, sess, train_data, validation_data):
    accuracy_graph = []
    y_ = tf.placeholder('float', [None, 30])

    #Define the cost function
    loss = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(y_, y))))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(loss)

    # saving and loading networks
    saver = tf.train.Saver()

    sess.run(tf.initialize_all_variables())

    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    #Create validation data
    x_v, y_v = get_sample(validation_data, len(validation_data.index)-1)
    #Create minimum loss profile
    min_loss = 99.0
    min_loss_rounds = 0

    for i in range(TRAIN_STEPS):
        x_s, y_s = get_sample(train_data, BATCH_SIZE)
        train_step.run(feed_dict={x: x_s, y_: y_s})
        if i % 5 == 0:
            i_loss = loss.eval(feed_dict={x: x_s, y_: y_s})
            validation_loss = loss.eval(feed_dict={x: x_v, y_: y_v})
            print('Step: %s Accuracy: %s Validation: %s'%(i, i_loss, validation_loss))
            accuracy_graph.append(i_loss)
        if validation_loss < min_loss:
            min_loss = validation_loss
            min_loss_rounds = 0
        else:
            min_loss_rounds += 1
        if min_loss_rounds > EARLY_STOPPING:
            saver.save(sess, 'saved_networks/facialKeypoints-dqn-early_stop', global_step=i)
            break
        if i%100 == 0:
            x_i, y_i = get_sample(validation_data, 1)
            markers = y.eval(feed_dict={x: x_i, y_: y_i})
            plot_img(x_i, markers, y_i)
            saver.save(sess, 'saved_networks/facialKeypoints-dqn', global_step=i)
    plt.plot(accuracy_graph)
    plt.show()



def main():
    data = pd.read_csv('/home/hazard/tf/FacialKeypoints/data/training.csv')
    data = convert_data(data)
    train_data, validation_data = train_test_split(data, test_size=VALIDATION_PERCENT, random_state=SEED)
    train_data = pd.DataFrame(data=train_data, columns=data.columns)
    validation_data = pd.DataFrame(data=validation_data, columns=data.columns)
    sess = tf.InteractiveSession()
    x, readout = create_network()
    train_network(x, readout, sess, train_data, validation_data)
    
    
    
    



if __name__ == '__main__':
    main()

