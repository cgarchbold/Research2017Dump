from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import Input
train_datagen = ImageDataGenerator()

test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
        '/u/amo-d0/guest/cohen/train_panos',
        target_size=(400, 300),
        batch_size=35,
        class_mode='sparse')
		
test_generator = test_datagen.flow_from_directory(
        '/u/amo-d0/guest/cohen/test_panos',
        target_size=(400, 300),
        batch_size=35,
        class_mode='sparse')


input_tensor = Input(shape=((53826/35),400, 300, 3))  # this assumes K.image_data_format() == 'channels_last'

base_model  = InceptionV3(weights='imagenet', include_top=False,)

x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
predictions = Dense(34, activation='softmax')(x)


# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

for layer in model.layers[:-10]:
    layer.trainable = False

	
model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
			  
model.summary()
model.fit_generator(
        train_generator,
        steps_per_epoch=10000,
        epochs=10,
		)

acc=model.evaluate_generator(self, test_generator, 1000, max_queue_size=10, workers=1, use_multiprocessing=False)
print(acc)