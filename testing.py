from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import Input
from PIL import ImageFile
from keras.models import load_model
ImageFile.LOAD_TRUNCATED_IMAGES = True
from keras.utils import plot_model
from keras.metrics import sparse_top_k_categorical_accuracy
from keras.metrics import sparse_categorical_accuracy
import keras.backend as K

train_datagen = ImageDataGenerator()

test_datagen = ImageDataGenerator()

valid_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
        'C:/Users/Cohen/Desktop/newtrain_panos',
        target_size=(400, 300),
        batch_size=30,
        class_mode='sparse')
		
test_generator = test_datagen.flow_from_directory(
        'C:/Users/Cohen/Desktop/test_panos',
        target_size=(400, 300),
        batch_size=30,
        class_mode='sparse')
		
valid_generator = valid_datagen.flow_from_directory(
        'C:/Users/Cohen/Desktop/valid_panos',
        target_size=(400, 300),
        batch_size=30,
        class_mode='sparse')

def top_5_categorical_accuracy(y_true, y_pred):
    return sparse_top_k_categorical_accuracy(y_true, y_pred, k=5) 
def top_10_categorical_accuracy(y_true, y_pred):
    return sparse_top_k_categorical_accuracy(y_true, y_pred, k=10) 
	
model = load_model('C:/Users/Cohen/Desktop/models/mymodeltwice.h5', custom_objects={"top_5_categorical_accuracy": top_5_categorical_accuracy})
model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy',top_5_categorical_accuracy,'sparse_categorical_accuracy',top_10_categorical_accuracy])
			  
print(model.metrics_names)
print(model.metrics_names)


accuracy =model.evaluate_generator(test_generator, 100, max_queue_size=100, workers=1, use_multiprocessing=False)
print(accuracy)