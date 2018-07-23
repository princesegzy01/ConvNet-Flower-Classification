from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import sys


def OneHotConverterResult(oneHotData):
    # define example
    #data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']

    data = [
        "Acer_Capillipes",
        "Acer_Circinatum",
        "Acer_Mono",
        "Acer_Opalus",
        "Acer_Palmatum",
        "Acer_Pictum",
        "Acer_Platanoids",
        "Acer_Rubrum",
        "Acer_Rufinerve",
        "Acer_Saccharinum",
        "Alnus_Cordata",
        "Alnus_Maximowiczii",
        "Alnus_Rubra",
        "Alnus_Sieboldiana",
        "Alnus_Viridis",
        "Arundinaria_Simonii",
        "Betula_Austrosinensis",
        "Betula_Pendula",
        "Callicarpa_Bodinieri",
        "Castanea_Sativa",
        "Celtis_Koraiensis",
        "Cercis_Siliquastrum",
        "Cornus_Chinensis",
        "Cornus_Controversa",
        "Cornus_Macrophylla",
        "Cotinus_Coggygria",
        "Crataegus_Monogyna",
        "Cytisus_Battandieri",
        "Eucalyptus_Glaucescens",
        "Eucalyptus_Neglecta",
        "Eucalyptus_Urnigera",
        "Fagus_Sylvatica",
        "Ginkgo_Biloba",
        "Ilex_Aquifolium",
        "Ilex_Cornuta",
        "Liquidambar_Styraciflua",
        "Liriodendron_Tulipifera",
        "Lithocarpus_Cleistocarpus",
        "Lithocarpus_Edulis",
        "Magnolia_Heptapeta",
        "Magnolia_Salicifolia",
        "Morus_Nigra",
        "Olea_Europaea",
        "Phildelphus",
        "Populus_Adenopoda",
        "Populus_Grandidentata",
        "Populus_Nigra",
        "Prunus_Avium",
        "Prunus_X_Shmittii",
        "Pterocarya_Stenoptera",
        "Quercus_Afares",
        "Quercus_Agrifolia",
        "Quercus_Alnifolia",
        "Quercus_Brantii",
        "Quercus_Canariensis",
        "Quercus_Castaneifolia",
        "Quercus_Cerris",
        "Quercus_Chrysolepis",
        "Quercus_Coccifera",
        "Quercus_Coccinea",
        "Quercus_Crassifolia",
        "Quercus_Crassipes",
        "Quercus_Dolicholepis",
        "Quercus_Ellipsoidalis",
        "Quercus_Greggii",
        "Quercus_Hartwissiana",
        "Quercus_Ilex",
        "Quercus_Imbricaria",
        "Quercus_Infectoria_sub",
        "Quercus_Kewensis",
        "Quercus_Nigra",
        "Quercus_Palustris",
        "Quercus_Phellos",
        "Quercus_Phillyraeoides",
        "Quercus_Pontica",
        "Quercus_Pubescens",
        "Quercus_Pyrenaica",
        "Quercus_Rhysophylla",
        "Quercus_Rubra",
        "Quercus_Semecarpifolia",
        "Quercus_Shumardii",
        "Quercus_Suber",
        "Quercus_Texana",
        "Quercus_Trojana",
        "Quercus_Variabilis",
        "Quercus_Vulcanica",
        "Quercus_x_Hispanica",
        "Quercus_x_Turneri",
        "Rhododendron_x_Russellianum",
        "Salix_Fragilis",
        "Salix_Intergra",
        "Sorbus_Aria",
        "Tilia_Oliveri",
        "Tilia_Platyphyllos",
        "Tilia_Tomentosa",
        "Ulmus_Bergmanniana",
        "Viburnum_Tinus",
        "Viburnum_x_Rhytidophylloides",
        "Zelkova_Serrata"
    ]


    values = array(data)
    #print(values)
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    #print(integer_encoded)

    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)


    #print("encoded")
    #print(onehot_encoded)
    # invert first example
    #inverted = label_encoder.inverse_transform([argmax(onehot_encoded[1, :])])
    inverted = label_encoder.inverse_transform([argmax(oneHotData)])

    #print("Inverted")
    #print(inverted)
    #sys.exit(0)

    return inverted




import tensorflow as tf
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.models import Sequential
from keras.layers.core import Activation
from keras.layers import Flatten
from keras.layers import Dense
import os
import sys




classifier = Sequential()

classifier.add(Conv2D(32, (3,3), input_shape=(128,128, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Flatten())

classifier.add(Dense(units = 256, activation = 'relu'))

classifier.add(Dense(units = 99, activation = 'softmax'))

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#using imageDataGenerator to preprocess our data
from keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('dataset/training',target_size=(128, 128), batch_size=32, class_mode='categorical')
test_generator = test_datagen.flow_from_directory('dataset/test', target_size=(128, 128), batch_size=32, class_mode='categorical')

#fit generate our model
classifier.fit_generator(train_generator, steps_per_epoch=200, epochs=10, validation_data=test_generator, validation_steps = 100)

print("Train generator indices : ", train_generator.class_indices)


classifier.save("model.h5")


print("done")
sys.exit(0)
#
from keras.preprocessing import image
import numpy as np

sn  = 0

print("================================================================================================================")

for img in sorted(os.listdir("dataset/test")):
    test_image = image.load_img("dataset/test/" + img, target_size = (128, 128))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)

    label = OneHotConverterResult(result)
    
    print(sn, " -- ", img, " -- ", label[0])
