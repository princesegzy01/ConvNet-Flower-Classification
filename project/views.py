from . import app
from flask import render_template
from flask import request
import os
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image as KerasImage
from keras.applications import imagenet_utils
from PIL import Image
import io
import numpy as np
from keras import backend as k
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from numpy import array
from numpy import argmax

global model
model = load_model("model.h5")

global graph
graph = tf.get_default_graph()

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']

    result = predict(file)
    return "<b>Flower Class is : </b>"  + result
    #return render_template('index.html')
def predict(img_path) :
    img = KerasImage.load_img( img_path , target_size = (128, 128))
    test_image = KerasImage.img_to_array(img)
    test_image = np.expand_dims(test_image, axis = 0)

    #print(test_image)
    flower_class = ""
    with graph.as_default():
	    #pred = model.predict(test_image)
        pred = model.predict(test_image)
        className = OneHotConverterResult(pred)
        #print(className) 

        flower_class = className[0]

    return flower_class


def OneHotConverterResult(oneHotData):
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
    onehot_encoder.fit_transform(integer_encoded)

    inverted = label_encoder.inverse_transform([argmax(oneHotData)])
    return inverted

