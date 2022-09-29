
# Uncomment for timing
# import time
# start = time.time()

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#from plot import plot
#from tqdm import tqdm

training_data = np.load('./data_prog4Spring22/videoframes_clips_train.npy')
training_label = np.load('./data_prog4Spring22/joint_3d_clips_train.npy')
testing_data = np.load('./data_prog4Spring22/videoframes_clips_valid.npy')
testing_label = np.load('./data_prog4Spring22/joint_3d_clips_valid.npy')


print(training_data.shape, training_label.shape)

# hyper-parameter
batch_size = 2
EPOCHS = 3
# test_dataset = (
#     tf.data.Dataset.from_tensor_slices((testing_data, testing_label)).batch(batch_size)
# )

# test_dataset = (
#     test_dataset.map(lambda x, y:
#                       (tf.divide(tf.cast(x, tf.float32), 255.0),tf.cast(y, tf.float32))))

def preprocess_data(data):
    return tf.divide(tf.cast(data, dtype=tf.float32), 255.) 


# Retrieves saved model if load = True
# Else, creates new Deep Dynamic Model
def getModel(load_file=None, lr=0.001):
    if load_file is not None:
        new_model = tf.keras.models.load_model(load_file, custom_objects={"MPJPE": MPJPE})
        return new_model

    model = DeepDynamicModel()
    # Adagrad and Adadelta converges later, run with many more iterations (maybe 200, maybe load existing model and continue training)
    model.compile(
        optimizer=tf.keras.optimizers.Adamax(learning_rate=lr),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[MPJPE()]
    )
    return model

#########################################################################################
# tf.keras subclass model
class MPJPE(tf.keras.metrics.Metric):
    def __init__(self):
        super(MPJPE, self).__init__()
        self.mpjpe = self.add_weight(name='mpjpe')

    def update_state(self, true_val, predict_val, sample_weight=None):
        self.mpjpe = tf.math.reduce_mean(tf.math.reduce_euclidean_norm((true_val - predict_val), axis=3)) * 1000

    def result(self):
        return self.mpjpe

class ResNetIdentity(tf.keras.Model):
    def __init__(self, filters):
        super(ResNetIdentity, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')
        self.batch_norm1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')
        self.batch_norm2 = tf.keras.layers.BatchNormalization()

        self.add_layer = tf.keras.layers.Add()
        self.activation = tf.keras.layers.Activation('relu')

    def call(self, input_data):
        input_skip = input_data

        output = self.conv1(input_data)
        output = self.batch_norm1(output)
        output = self.activation(output)
        output = self.conv2(output)
        output = self.batch_norm2(output)
        output = self.add_layer([output, input_skip])
        output = self.activation(output)

        return output

class ResNetConvolution(tf.keras.Model):
    def __init__(self, filters):
        super(ResNetConvolution, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', strides=(2, 2))
        self.batch_norm1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')
        self.batch_norm2 = tf.keras.layers.BatchNormalization()

        self.conv3 = tf.keras.layers.Conv2D(filters, (1,1), strides=(2,2))

        self.add_layer = tf.keras.layers.Add()
        self.activation = tf.keras.layers.Activation('relu')

    def call(self, input_data):
        input_skip = input_data
        
        output = self.conv1(input_data)
        output = self.batch_norm1(output)
        output = self.activation(output)
        output = self.conv2(output)
        output = self.batch_norm2(output)
        input_skip = self.conv3(input_skip)
        output = self.add_layer([output, input_skip])
        output = self.activation(output)

        return output

class ResNet(tf.keras.Model):
    def __init__(self):
        super(ResNet, self).__init__()
        self.zero_padding = tf.keras.layers.ZeroPadding3D((0, 3, 3))

        self.conv = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same')
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.Activation('relu')
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')

        self.identity_block1 = ResNetIdentity(64)
        self.identity_block2 = ResNetIdentity(64)

        self.convolutional_block1 = ResNetConvolution(128)
        self.identity_block3 = ResNetIdentity(128)

        self.convolutional_block2 = ResNetConvolution(256)
        self.identity_block4 = ResNetIdentity(256)

        self.convolutional_block3 = ResNetConvolution(512)
        self.identity_block5 = ResNetIdentity(512)

        self.average_pool = tf.keras.layers.AveragePooling2D((2, 2), padding='same')
        self.flatten = tf.keras.layers.Flatten()
        self.output_layer = tf.keras.layers.Dense(512, activation='relu')

    def call(self, input_data):
        #output = self.zero_padding(input_data)
        #print(output.shape)
        output = self.conv(input_data)
        output = self.batch_norm(output)
        output = self.activation(output)
        #print(output.shape)
        output = tf.reshape(output, (tf.shape(output)[0]*tf.shape(output)[1], tf.shape(output)[2], tf.shape(output)[3], tf.shape(output)[4]))
        output = self.max_pool(output)
        output = tf.reshape(output, (tf.cast(tf.shape(output)[0]/8, dtype=tf.int32), 8, tf.shape(output)[1], tf.shape(output)[2], tf.shape(output)[3]))
        output = self.identity_block1(output)
        output = self.identity_block2(output)
        output = self.convolutional_block1(output)
        output = self.identity_block3(output)
        output = self.convolutional_block2(output)
        output = self.identity_block4(output)
        output = self.convolutional_block3(output)
        output = self.identity_block5(output)
        output = tf.reshape(output, (tf.shape(output)[0]*tf.shape(output)[1], tf.shape(output)[2], tf.shape(output)[3], tf.shape(output)[4]))
        output = self.average_pool(output)
        #output = self.flatten(output)
        output = tf.reshape(output, (tf.cast(tf.shape(output)[0]/8, dtype=tf.int32), 8, tf.shape(output)[1]*tf.shape(output)[2]*tf.shape(output)[3]))
        # output = tf.map_fn(self.average_pool, output)
        # output = tf.map_fn(self.flatten, output)
        output = self.output_layer(output)

        return output

class DeepDynamicModel(tf.keras.Model):
    def __init__(self):
        super(DeepDynamicModel, self).__init__()
        self.res_net = ResNet()
        self.lstm = tf.keras.layers.LSTM(1024, return_sequences=True, dropout=0.2)
        self.dense1 = tf.keras.layers.Dense(500, activation='relu')
        self.dense2 = tf.keras.layers.Dense(200, activation='relu')
        self.output_layer = tf.keras.layers.Dense(17*3)

    def call(self, input_data, training=False):
        # total_output = np.empty((input_data.shape[0], input_data.shape[1], 17, 3))
        # for i in range(8):
        #     temp_data = input_data[:, i]
        output = self.res_net(input_data)
        output = self.lstm(output, training=training)
        output = self.dense1(output)
        output = self.dense2(output)
        output = self.output_layer(output)
        output = tf.reshape(output, (tf.shape(output)[0], tf.shape(output)[1], 17, 3))

        return output

        # total_output[:, i] = output[:, i]

        # return tf.convert_to_tensor(total_output, dtype=tf.float32)

#########################################################################################

# Plots provided loss and accuracy over e epochs
def plotLossAccuracy(loss, acc, e, name):
    epochs = list(range(e))

    plt.figure(1)
    plt.plot(epochs, loss, color='green')
    plt.title(name + " Loss vs. Epochs")
    plt.xlabel("epochs")
    plt.ylabel(name + " Loss")
    plt.savefig(name + ' Loss.png')

    plt.figure(2)
    plt.plot(epochs, acc, color='green')
    plt.title(name + " Error vs. Epochs")
    plt.xlabel("epochs")
    plt.ylabel(name + " Error")
    plt.savefig(name + ' Error.png')
    plt.show()


if __name__ == '__main__':
    # The following implementation is for students who construct model with tf.keras subclass model
    # Please adjust the following part accordingly if you adopt other implementation method.
    train_data = preprocess_data(training_data)
    test_data = preprocess_data(testing_data)

    # end = time.time()

    # print("Preprocessing complete:", end - start)

    # load_file='trained_model', 
    model = getModel(lr=0.0001)
    # , validation_data=(test_data, testing_label)
    use_weights = True
    if not use_weights:
        history = model.fit(train_data, training_label, batch_size=batch_size, epochs=EPOCHS, validation_data=(test_data, testing_label))
        model.save_weights('model_weights')

        # Record and plot loss and accuracy
        # train_loss = history.history['loss']
        # train_MJPJE = history.history['mpjpe']
        # plotLossAccuracy(train_loss, train_MJPJE, EPOCHS, "Training")

        # # Record and plot test loss and accuracy
        # test_loss = history.history['val_loss']
        # test_MJPJE = history.history['val_mpjpe']
        # plotLossAccuracy(test_loss, test_MJPJE, EPOCHS, "Testing")
    else:
        model.build((1, 8, 224, 224, 3))
        model.load_weights('model_weights')
    
    predict = model.predict(test_data, batch_size=batch_size)


    MPJPE = tf.math.reduce_mean(tf.math.reduce_euclidean_norm((testing_label - predict), axis=3)) * 1000
    print('Final MPJPE:', MPJPE.numpy())

    # Evaluate model on test data and plot loss and accuracy
    # Also record classification error for each class and average classification error
    # test_loss = history.history['val_loss']
    # test_acc = history.history['val_categorical_accuracy']
    # # print('Model accuracy: {:5.2f}%'.format(100 * test_acc))
    # plotLossAccuracy(test_loss, test_acc, EPOCHS, "Testing")



    # initialize
    # model.build((1, 8, 224, 224, 3))
    # load model
    # model.load_weights('model_weights')
    # predict = model.predict(test_data, batch_size=batch_size)
    # MPJPE = tf.math.reduce_mean(tf.math.reduce_euclidean_norm((testing_label - predict), axis=3)) * 1000
    # print('Final MPJPE:', MPJPE.numpy())

    # tmp_MPJPE = []
    # tmp_loss = []
    # for x, y in test_dataset:
    #     predict = model.predict(x)
    #     # F norm(y-predict) ^ 2 / batch_size
    #     loss = tf.math.reduce_sum(tf.losses.mean_squared_error(y, predict)) / batch_size
    #     # batch_size x frame x 17 x 1
    #     MPJPE = tf.math.reduce_mean(tf.math.reduce_euclidean_norm((y - predict), axis=3)) * 1000
    #     tmp_MPJPE.append(MPJPE)
    #     tmp_loss.append(loss)
    # testing_los = tf.reduce_mean(tmp_loss)
    # test_MPJPE = tf.reduce_mean(tmp_MPJPE)
    # print('MPJPE:', test_MPJPE.numpy())


