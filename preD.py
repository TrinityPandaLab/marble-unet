#use only
from model import *
from data import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

Path = 'test'
model_mode = 'sawdust'
image_name = '103.tif'
model_Name = 'unet_'+model_mode+'.hdf5'
model = load_model(os.path.join(Path,model_Name))



img = io.imread(os.path.join(Path,image_name),as_gray = False)
img = trans.resize(img,[256,256])
img = np.reshape(img,(1,)+img.shape)

results = model.predict(img,1,verbose=1)
#saveResult(Path,results)
img = results[0,:,:]
print(results.shape)
io.imsave(os.path.join(Path,"result.png"),img)
