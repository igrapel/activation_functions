#Imports
import tensorflow as tf
import keras
import math

from tensorflow.keras import models, layers
from tensorflow.keras.models import Sequential

import numpy as np

from art.attacks.evasion import FastGradientMethod, CarliniL2Method, CarliniLInfMethod, BoundaryAttack, DeepFool
from art.estimators.classification import TensorFlowV2Classifier

from art.utils import load_dataset

#import MNIST
print("Finished Imports . . . ")
print("Starting to import MNIST")
from art.utils import load_dataset
(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset(str("mnist"))
