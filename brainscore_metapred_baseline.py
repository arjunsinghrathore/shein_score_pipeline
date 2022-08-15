import functools
import tensorflow as tf
import cv2
from model_tools.activations.pytorch import load_preprocess_images

############################################################
############################################################


_mean_imagenet = tf.constant([0.485, 0.456, 0.406], shape=[1, 1, 3], dtype=tf.float32)
_std_imagenet =  tf.constant([0.229, 0.224, 0.225], shape=[1, 1, 3], dtype=tf.float32)

def load_images(image_filepaths,image_size):
    
    return np.array([load_image(image_filepath,image_size) for image_filepath in image_filepaths])

def load_image(image_filepath,image_size):
    
    original_image = cv2.imread(image_filepath)
    height, width = original_image.shape[:2]
    
    if len(original_image.shape)==2:
        original_image = gray2rgb(original_image)
    #image = transform_gen.get_transform(original_image).apply_image(original_image)
    
    image = tf.image.resize(original_image,(image_size,image_size)).numpy()
    image = tf.cast(image, tf.float32)/255.0
    image -= _mean_imagenet
    image /= _std_imagenet
    
    #inputs = {"image": image, "height": height, "width": width}
        
    return image

def load_preprocess_images(image_filepaths, image_size=256,**kwargs):
    #torch.cuda.empty_cache()
    images = load_images(image_filepaths,image_size)
    return images

preprocessing = functools.partial(load_preprocess_images, image_size=224)


############################################################
############################################################

from model_tools.activations.pytorch import PytorchWrapper
from model_tools.activations.keras import KerasWrapper

#activations_model = PytorchWrapper(identifier='my-model', model=MyModel(), preprocessing=preprocessing)

############################################################
############################################################

import glob 

models_eff =glob.glob('/media/data_cifs/projects/prj_metapredictor/meta_models/models/eff*h5')

print('models_eff : ',models_eff)

############################################################
############################################################

models =['/media/data_cifs/projects/prj_metapredictor/meta_models/models/vgg_baseline.h5',
          '/media/data_cifs/projects/prj_metapredictor/meta_models/models/vgg_frosty_eon.h5',
 '/media/data_cifs/projects/prj_metapredictor/meta_models/models/resnet50_baseline.h5',
         '/media/data_cifs/projects/prj_metapredictor/meta_models/models/saliency_volcanic_monkey.h5',
 '/media/data_cifs/projects/prj_metapredictor/meta_models/models/vgg_silver_moon.h5',
         ]

############################################################
############################################################

from model_tools.activations.core import ActivationsExtractorHelper
import numpy as np
from collections import OrderedDict
class Tensorflow2Wrapper:
    def __init__(self, identifier,model,preprocessing, *args, **kwargs):
        import tensorflow as tf
        self._model = model
        
        self._extractor = ActivationsExtractorHelper(identifier=identifier, get_activations=self.get_activations,
                                                     preprocessing=preprocessing, *args, **kwargs)
        self._extractor.insert_attrs(self)

    @property
    def identifier(self):
        return self._extractor.identifier

    @identifier.setter
    def identifier(self, value):
        self._extractor.identifier = value

    def __call__(self, *args, **kwargs):  # cannot assign __call__ as attribute due to Python convention
        return self._extractor(*args, **kwargs)
    def batch_predict(self, inputs, batch_size=10):
        activations = []
        for batch_x in tf.data.Dataset.from_tensor_slices(inputs).batch(batch_size):
            #print('here')
            batch_act = self._model(batch_x)
            batch_act = np.array(batch_act, np.float16)
            activations += list(batch_act)
        return np.array(activations, np.float16)
    
    def get_activations(self, images, layer_names):
        layer_outputs = []
        for layer in layer_names: 
            activation_model = tf.keras.Model(self._model.input, self._model.get_layer(layer).output)
            #import pdb;pdb.set_trace()
            layer_outputs.append(self.batch_predict(images))  # 0 to signal testing phase
        return OrderedDict([(layer_name, layer_output) for layer_name, layer_output in zip(layer_names, layer_outputs)])
        

############################################################
############################################################

import efficientnet 

efficientnet.init_tfkeras_custom_objects()

############################################################
############################################################

from brainscore import score_model
import pandas as pd
from model_tools.brain_transformation import ModelCommitment
results = []
for MODEL_NAME in models_eff:
    print(MODEL_NAME)
    model = tf.keras.models.load_model(MODEL_NAME,compile=False)
    layers = [n.name for n in model.layers[-10:]]
    activations_model = Tensorflow2Wrapper(identifier=MODEL_NAME, model=model, preprocessing=preprocessing)
    model = ModelCommitment(identifier=MODEL_NAME, activations_model=activations_model,layers = layers )
    score_v4 = score_model(model_identifier=MODEL_NAME, model=model,
                    benchmark_identifier='dicarlo.MajajHong2015public.V4-pls')
    print(score_v4)
    score_it = score_model(model_identifier=MODEL_NAME, model=model,
                    benchmark_identifier='dicarlo.MajajHong2015public.IT-pls',verbose=0)
    print(score_it)
    
    results.append([MODEL_NAME,score_it,score_v4])
    print(results)
    rdf = pd.DataFrame(results,columns=['model','score_it','score_v4'])
    rdf.to_csv('bs_score_our_models_eff.csv')

############################################################
############################################################

print('rdf.iloc[0].keys() : ',rdf.iloc[0].keys())

parse = []
for i in range(len(rdf)):
    row =  rdf.iloc[i]
    model = row['model'].split('/')[-1]
    raw_it_m = row['score_it'].raw.values[0]
    raw_it_s = row['score_it'].raw.values[1]
    raw_v4_m = row['score_v4'].raw.values[0]
    raw_v4_s = row['score_v4'].raw.values[1]
    ceiling_it_m = row['score_it'].ceiling.values[0]
    ceiling_it_s = row['score_it'].ceiling.values[1]
    ceiling_v4_m = row['score_v4'].ceiling.values[0]
    ceiling_v4_s = row['score_v4'].ceiling.values[1]
    parse.append([model,raw_it_m,raw_it_s,raw_v4_m,raw_v4_s,ceiling_it_m,ceiling_it_s,ceiling_v4_m,ceiling_v4_s])


parsedf_eff = pd.DataFrame(parse,columns=['model','raw_it_m','raw_it_s','raw_v4_m','raw_v4_s','ceiling_it_m','ceiling_it_s','ceiling_v4_m','ceiling_v4_s'])