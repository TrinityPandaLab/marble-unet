#use only
from model import *
from data import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


model = load_model('unet_marble.hdf5')


testGene = resultGenerator("datamy/marbles/test")
results = model.predict(testGene,526,verbose=1)
saveResult("datamy/marbles/test",results)
