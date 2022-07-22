#Imports
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.models import Sequential

import numpy as np

from art.attacks.evasion import FastGradientMethod, CarliniL2Method, CarliniLInfMethod, BoundaryAttack, DeepFool
from art.estimators.classification import TensorFlowV2Classifier

import keras
from scipy.special import gamma

#import MNIST
print("Finished Imports ...........")
print("Starting to import MNIST ")
from art.utils import load_dataset
(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset(str("mnist"))
print("Finished Importing MNIST")

sess = tf.compat.v1.Session()
print(sess)
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
  
print("Defined the model, Fast Gradient Sign Method, boundary attack, and deepfool attack.........................................")

# Evaluate Function
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

# '''
# HYPERPARAMETERS
# '''
a = 2 # width of the barrier
#
# # STATIC
m = 1 # mass
hbar = 1 # reduced Plank's constant
V = 1 # potential barrier


print("Load Delta Dirac Tests")
def k(E): # k1 function to relate wave values to the energy 
    return tf.math.sqrt(2*m*tf.math.abs(E-V)) / hbar**2

def T_square(E): # quantum tunneling with the square barrier
    e_less_than_v = 1 / (1 + (tf.math.sinh(k(E)*a))**2 / (4*E*tf.math.abs(V-E)))
    e_greater_than_v = 1 / (1 + (tf.math.sin(k(E)*a))**2 / (4*E*tf.math.abs(V-E)))
    e_equal_v = 1 / (1 + a**2/2)
    
    not_equal = tf.where(E < V, e_less_than_v, e_greater_than_v)
    return tf.where(E == V, e_equal_v, not_equal)

def T_square_derivative(x):
    e_less_than_v = -((4*tf.math.sinh(a*tf.math.sqrt(2-2*x))*(tf.math.sqrt(1-x)*(2*x-1)*tf.math.sinh(a*tf.math.sqrt(2-2*x)) + 1.4142*a*(x-1)*x*tf.math.cosh(a*tf.math.sqrt(2-2*x)))) / (tf.math.sqrt(1-x)*(tf.math.sinh(a*tf.math.sqrt(2-2*x))**2 - 4*(x-1)*x)**2))
    e_greater_than_v = (4*tf.math.sin(1.4142*a*tf.math.sqrt(x-1))*(tf.math.sqrt(x-1)*(2*x-1)*tf.math.sin(1.4142*a*tf.math.sqrt(x-1)) - 1.4142*a*(x-1)*x*tf.math.cos(1.4142*a*tf.math.sqrt(x-1)))) / (tf.math.sqrt(x-1)*(tf.math.sin(1.4142*a*tf.math.sqrt(x-1))**2 + 4*(x-1)*x)**2)
    e_equal_v = 0
    
    not_equal = tf.where(x < V, e_less_than_v, e_greater_than_v)
    return tf.where(x == V, e_equal_v, not_equal)

@tf.custom_gradient
def T_square_activation(x):

    def grad(dy):
        return T_square_derivative(x) * dy

    result = T_square(x)
    return result, grad
print('value of a used in the activation: ', a)

#Dirac Classifier
model = define_model(T_square_activation)
classifier = train_model(model, x_train, y_train, x_test, y_test)

eval_x_test, eval_y_test = get_successful_test(classifier, x_test, y_test)

print("Finished Dirac Classifier Model")

for epsilon in [0.02, 0.04, 0.06, 0.2, 0.4]:
    fgsm_attack(classifier, eval_x_test, eval_y_test, eps=epsilon)
    
boundary_attack(classifier, eval_x_test, eval_y_test)
print("End of boundary Attack")
print("Start of DeepFool Attack")
deepfool_attack(classifier, eval_x_test, eval_y_test)


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

print("Created Gamma Function ...........................................")

# Test
print("Testing ......")
model = define_model(gamma_activation)
classifier = train_model(model, x_train, y_train, x_test, y_test, eps=15)

eval_x_test, eval_y_test = get_successful_test(classifier, x_test, y_test)

print("Finished with Gamma classifier")

for epsilon in [0.02, 0.04, 0.06, 0.2, 0.4]:
    fgsm_attack(classifier, eval_x_test, eval_y_test, eps=epsilon)

boundary_attack(classifier, eval_x_test, eval_y_test)
deepfool_attack(classifier, eval_x_test, eval_y_test)

print("End of Gamma Tests")
