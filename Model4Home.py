from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import Input
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from keras.metrics import sparse_top_k_categorical_accuracy
from keras.metrics import sparse_categorical_accuracy
 
train_datagen = ImageDataGenerator()

test_datagen = ImageDataGenerator()

valid_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
        'C:/Users/Cohen/Desktop/newtrain_panos',
        target_size=(400, 300),
        batch_size=15,
        class_mode='sparse')
		
test_generator = test_datagen.flow_from_directory(
        'C:/Users/Cohen/Desktop/test_panos',
        target_size=(400, 300),
        batch_size=15,
        class_mode='sparse')
		
valid_generator = valid_datagen.flow_from_directory(
        'C:/Users/Cohen/Desktop/valid_panos',
        target_size=(400, 300),
        batch_size=15,
        class_mode='sparse')


input_tensor = Input(shape=(15,400, 300, 3))  # this assumes K.image_data_format() == 'channels_last'

base_model  = InceptionV3(weights='imagenet', include_top=False,)

x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
predictions = Dense(33, activation='softmax')(x)


# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

for layer in model.layers[:10]:
    layer.trainable = False
	
def top_5(y_true, y_pred):
    return sparse_top_k_categorical_accuracy(y_true, y_pred, k=5) 
	
model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=[sparse_categorical_accuracy,top_5])
			  
model.summary()
model.fit_generator(
        train_generator,
        steps_per_epoch=1000,
        epochs=10,
		validation_data=valid_generator,
		validation_steps=100
		)
		
model.save('C:/Users/Cohen/Desktop/models/mymodel.h5')
acc=model.evaluate_generator(test_generator, 1000, max_queue_size=10, workers=1, use_multiprocessing=False)

print(acc)