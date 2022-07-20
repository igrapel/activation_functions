#Imports
import tensorflow as tf
import keras
import math

from tensorflow.keras import models, layers
from tensorflow.keras.models import Sequential

from scipy import stats
from scipy.special import gamma

import numpy as np

from art.attacks.evasion import FastGradientMethod, CarliniL2Method, CarliniLInfMethod, BoundaryAttack, DeepFool
from art.estimators.classification import TensorFlowV2Classifier

from art.utils import load_dataset

#import MNIST
print("Finished Imports ...........")
print("Starting to import MNIST ")
from art.utils import load_dataset
(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset(str("mnist"))
print("Finished Importing MNIST")

#Create Model
def define_model(af=None): 
    model = Sequential()

    # C1 convolutional layer 
#     model.add(layers.Conv2D(6, kernel_size=(5,5), strides=(1,1), activation='tanh', input_shape=(28,28,1), padding='same'))
    if af == None: 
        model.add(layers.Conv2D(6, kernel_size=(5,5), strides=(1,1), activation='tanh', padding='same'))
    else: 
        model.add(layers.Conv2D(6, kernel_size=(5,5), strides=(1,1), activation=af, padding='same', dynamic=True))

    # S2 pooling layer
    model.add(layers.AveragePooling2D(pool_size=(2,2), strides=(1,1), padding='valid'))

    # C3 convolutional layer
#     model.add(layers.Conv2D(16, kernel_size=(5,5), strides=(1,1), activation='tanh', padding='valid'))
    if af == None: 
        model.add(layers.Conv2D(16, kernel_size=(5,5), strides=(1,1), activation='tanh', padding='valid'))
    else: 
        model.add(layers.Conv2D(16, kernel_size=(5,5), strides=(1,1), activation=af, padding='valid', dynamic=True))

    # S4 pooling layer
    model.add(layers.AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

    # C5 fully connected convolutional layer
    if af == None: 
        model.add(layers.Conv2D(120, kernel_size=(5,5), strides=(1,1), activation='tanh', padding='valid'))
    else: 
        model.add(layers.Conv2D(120, kernel_size=(5,5), strides=(1,1), activation=af, padding='valid', dynamic=True))
    model.add(layers.Flatten())

    # FC6 fully connected layer
    if af == None: 
        model.add(layers.Dense(84, activation='tanh'))
    else: 
        model.add(layers.Dense(84, activation=af, dynamic=True))

    # Output layer with softmax activation
    model.add(layers.Dense(10, activation='softmax'))
    return model

def train_step(model, images, labels):
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
        
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
def train_model(model, x_train, y_train, x_test, y_test, eps=10, batch=128, lr=0.01, filename=None):
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    classifier = TensorFlowV2Classifier(model=model,
                                        clip_values=(min_, max_), 
                                        input_shape=x_train.shape[1:], 
                                        nb_classes=10,  
                                        train_step=train_step,
                                        loss_object=loss_object)
    print('...created classifier')
    hist = classifier.fit(x_train, y_train, nb_epochs=eps, batch_size=batch)
    print('...finished training')
    
    # # Evaluate the classifier on the test set
    preds = np.argmax(classifier.predict(x_test), axis=1)
    acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
    print("Test accuracy: %.2f%%\n" % (acc * 100))
    
    loss = classifier.compute_loss(x_train, y_train, training_mode=True)
    print('Training loss: ', loss)

    return classifier
  
print("Finished creating the model")
#Adversarial Attacks
def fgsm_attack(classifier, x_test, y_test, eps=0.2):
    epsilon = eps  # Maximum perturbation
    adv_crafter = FastGradientMethod(classifier, eps=epsilon)
    print('...creating adversarial examples')
    x_test_adv = adv_crafter.generate(x=x_test)

    # Evaluate the classifier on the adversarial examples
    preds = np.argmax(classifier.predict(x_test_adv), axis=1)
    acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
    print("Test accuracy on adversarial sample: %.2f%%" % (acc * 100))
    
def boundary_attack(classifier, x_test, y_test):
    adv_crafter = BoundaryAttack(classifier, targeted=False, max_iter=0, delta=0.001, epsilon=0.001, init_size=5)
    print('...creating adversarial examples')
    x_test_adv = adv_crafter.generate(x=x_test, y=y_test)

    # Evaluate the classifier on the adversarial examples
    preds = np.argmax(classifier.predict(x_test_adv), axis=1)
    acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
    print("Test accuracy on adversarial sample: %.2f%%" % (acc * 100))
    
def deepfool_attack(classifier, x_test, y_test):
    adv_crafter = DeepFool(classifier)
    print('...creating adversarial examples')
    x_test_adv = adv_crafter.generate(x=x_test)

    # Evaluate the classifier on the adversarial examples
    preds = np.argmax(classifier.predict(x_test_adv), axis=1)
    acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
    print("Test accuracy on adversarial sample: %.2f%%" % (acc * 100))

def get_successful_test(classifier, x_test, y_test):
    preds = np.argmax(classifier.predict(x_test), axis=1)
    acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
    print("Original test accuracy: %.2f%%" % (acc * 100))
    
    preds = np.argmax(classifier.predict(x_test), axis=1)
    correct = np.nonzero(preds == np.argmax(y_test, axis=1))

    eval_x_test = x_test[correct]
    eval_y_test = y_test[correct]

    eval_x_test_final = eval_x_test[:1000]
    print(eval_x_test_final.shape)
    eval_y_test_final = eval_y_test[:1000]
    print(eval_y_test_final.shape)
    
    preds = np.argmax(classifier.predict(eval_x_test_final), axis=1)
    acc = np.sum(preds == np.argmax(eval_y_test_final, axis=1)) / eval_y_test_final.shape[0]
    print("Test set of correctly predicted (benign): %.2f%%" % (acc * 100))
    
    return eval_x_test_final, eval_y_test_final
  
print("Defined the model, Fast Gradient Sign Method, boundary attack, and deepfool attack........................................................")

# Create the Generalized Gamma Function
'''
HYPERPARAMETERS
'''
a = 1 # alpha
b = 3 # beta
c = 3 # gamma
mu = -2.61  # mu
sf = 1.17 # scale factor

'''
FUNCTIONS
'''
def generalized_gamma(x):
    x = tf.math.divide(x-mu, b)
    func = tf.math.divide(tf.math.exp(-x**c)*c*x**((c*a)-1), gamma(a))    
    return tf.where(x>0, tf.math.divide(func, sf), 0)
def gamma_derivative(x):
    x = tf.Variable(x, name='x')
    with tf.GradientTape(persistent=True) as tape: 
        y = tf.constant(generalized_gamma(x), dtype='float32')
    dy_dx = tape.gradient(y, x)
    return dy_dx

@tf.custom_gradient
def gamma_activation(x):
    def grad(dy):
        return gamma_derivative(x) * dy

    result = generalized_gamma(x)
    return result, grad
model = define_model(gamma_activation)
classifier = train_model(model, x_train, y_train, x_test, y_test, eps=15)

eval_x_test, eval_y_test = get_successful_test(classifier, x_test, y_test)

print("Created and Evaluated the Gamma Function ...........................................")
