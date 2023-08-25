from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications import VGG16

vgg_face = VGG16(weights = 'imagenet', include_top = False, input_shape = (224,224,3))

for layer in vgg_face.layers:
  layer.trainable = False
  
for (i,layer) in enumerate(vgg_face.layers):
    print(str(i) + " "+ layer.__class__.__name__, layer.trainable)
    

top = vgg_face.output
top = GlobalAveragePooling2D()(top)
top = Dense(1024, activation = 'relu')(top)
top = Dense(1024, activation = 'relu')(top)
top = Dense(512, activation = 'relu')(top)
top = Dense(5, activation = 'softmax')(top)

# 5 as Krish, Manish, Sarika, Sonya and unknown

model = Model(inputs = vgg_face.input, outputs = top)
print(model.summary())

train_dir = '/Users/krishaanggupta/Desktop/ML projects/trainingext'
validation_dir = '/Users/krishaanggupta/Desktop/ML projects/validationext'

train_dataset = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 45,
    width_shift_range = 0.3,
    height_shift_range = 0.3,
    horizontal_flip = True,
    fill_mode = 'nearest'
)

validation_dataset = ImageDataGenerator(
    rescale = 1./255
)


train_gen = train_dataset.flow_from_directory(
    train_dir,
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'categorical'
)
validation_gen = validation_dataset.flow_from_directory(
    validation_dir,
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'categorical'
)

check = ModelCheckpoint(
    "face_rec.h5",
    monitor = "val_loss",
    mode = "min",
    save_best_only = True,
    verbose = 1
)

earlystop = EarlyStopping(
    monitor = 'val_loss',
    min_delta = 0,
    patience = 3,
    verbose = 1,
    restore_best_weights = True
)
callbacks = [earlystop, check]

model.compile(
    loss = 'categorical_crossentropy',
    optimizer = RMSprop(lr = 0.001),
    metrics = ['accuracy']
)

nb_train = 180
nb_validation = 102

epochs = 10
batch_size = 32
history = model.fit(
    train_gen,
    steps_per_epoch = nb_train // batch_size,
    epochs = epochs,
    callbacks = callbacks,
    validation_data = validation_gen,
    validation_steps = nb_validation // batch_size
)