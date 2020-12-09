import keras
from keras.layers import Dense
from keras import Model
def get_baseline_model():
    # MODEL: BASELINE
    base_model = keras.applications.inception_v3.InceptionV3(include_top=True, weights='imagenet', input_shape=(224,224,3))  #Xception(include_top=True, weights='imagenet') #inception_v3.InceptionV3(include_top=True, weights='imagenet')
    # freezing some layers
    #layers_list=['conv2d_92', 'conv2d_93', 'conv2d_88', 'conv2d_89', 'conv2d_86']
    layers_list=[]
    for i in range(len(base_model.layers[:])):
        layer=base_model.layers[i]
        if layer.name in layers_list:
            print layer.name
            layer.trainable=True
        else:
            layer.trainable = False

    feature_output_first=base_model.layers[-2].output
    # adding dropout as extra-regularization
    dropout_layer = keras.layers.Dropout(0.8, noise_shape=None, seed=None)(feature_output_first)
    feature_output = Dense(2048, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01), name='finetuned_features1')(dropout_layer)
    dropout_layer = keras.layers.Dropout(0.8, noise_shape=None, seed=None)(feature_output)
    feature_output = Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01), name='finetuned_features2')(dropout_layer)
    dropout_layer = keras.layers.Dropout(0.8, noise_shape=None, seed=None)(feature_output)
    feature_output = Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01), name='finetuned_features3')(dropout_layer)

    finetuning = Dense(1, name='predictions')(feature_output)
    #finetuning = Dense(1, name='predictions', activation ='sigmoid')(feature_output)
    model = Model(input=base_model.input, output=finetuning)
    return model

def get_random_baseline_model():
    # MODEL: BASELINE
    base_model = keras.applications.inception_v3.InceptionV3(include_top=True, weights=None, input_shape=(224,224,3))  #Xception(include_top=True, weights='imagenet') #inception_v3.InceptionV3(include_top=True, weights='imagenet')
    # freezing some layers
    #layers_list=['conv2d_92', 'conv2d_93', 'conv2d_88', 'conv2d_89', 'conv2d_86']
    layers_list=[]
    for i in range(len(base_model.layers[:])):
        layer=base_model.layers[i]
        if layer.name in layers_list:
            print layer.name
            layer.trainable=True
        else:
            layer.trainable = False

    feature_output_first=base_model.layers[-2].output
    # adding dropout as extra-regularization
    dropout_layer = keras.layers.Dropout(0.8, noise_shape=None, seed=None)(feature_output_first)
    feature_output = Dense(2048, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01), name='finetuned_features1')(dropout_layer)
    dropout_layer = keras.layers.Dropout(0.8, noise_shape=None, seed=None)(feature_output)
    feature_output = Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01), name='finetuned_features2')(dropout_layer)
    dropout_layer = keras.layers.Dropout(0.8, noise_shape=None, seed=None)(feature_output)
    feature_output = Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01), name='finetuned_features3')(dropout_layer)

    finetuning = Dense(1, name='predictions')(feature_output)
    #finetuning = Dense(1, name='predictions', activation ='sigmoid')(feature_output)
    model = Model(input=base_model.input, output=finetuning)
    return model