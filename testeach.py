from model import *
from data import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

test_num = 52
kpath = 'Kfolder'
i = 0
myGene = trainGenerator(2,os.path.join(kpath,str(i),'train'),'image','labels',os.path.join(kpath,str(i),'train','aug'),data_gen_args,)
model = unet()
#model_checkpoint = ModelCheckpoint(os.path.join(kpath,str(i),'unet_marble_'+str(i)+'_.hdf5'), monitor='loss',verbose=1, save_best_only=True)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
history = model.fit(myGene,steps_per_epoch=10,epochs=2,callbacks=[early_stop])
length = len(history.history['loss'])
print(history.history['loss'][length-1])
testGene = testGenerator(os.path.join(kpath,str(i),'test'),num_image = test_num)
results = model.predict(testGene,test_num,verbose=1)
saveResult(i,os.path.join(kpath,str(i),'test'),results)
