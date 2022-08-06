from model import *
from data import *
import csv
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
k_value = 10
n_img = 526
batch_size = 1
i = 0
double_t = False
epochs = 10
steps_per_epoch = 100
start = time.time()
early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
if double_t:
    marble_path = os.path.join('Labels-double','marbles')
    sawdust_path = os.path.join('Labels-double','sawdust')
    #kfolderGenerator_double(k_value,'Labels','inputs',marble_path,sawdust_path)
    #-------------------------------------------------------------marble--------------------------------
    while i < k_value:
    #for i in range(k_value):
        print('Now working on folder -'+str(i))
        test_num = math.floor(n_img/k_value)
        if i == k_value-1:
            test_num += n_img%k_value
        kpath = 'Kfolder'
        #myGene = trainGenerator(batch_size,os.path.join(kpath,str(i),'train'),'image','labels',os.path.join(kpath,str(i),'train','aug'),data_gen_args,)
        marbleGene = trainGenerator(batch_size,os.path.join(kpath,str(i),'train'),'image','marble',None,data_gen_args)
        marble_model = unet()       
        marble_history = marble_model.fit(marbleGene,steps_per_epoch=steps_per_epoch,epochs=epochs,callbacks=[early_stop])
        marble_length = len(marble_history.history['loss'])
        testGene = testGenerator(os.path.join(kpath,str(i),'test'),num_image = test_num)
        results = marble_model.predict(testGene,test_num,verbose=1)
        print('is here')
        saveResult_double(i,True,os.path.join(kpath,str(i),'test'),results)
        loss_result = marble_history.history['loss']
        csvheader = 'This is fold '+str(i) + 'for marble'
        with open('loss_result.csv', 'a', newline ='') as f:
    	    write = csv.writer(f)
    	    write.writerow(csvheader)
    	    write.writerow(loss_result)
    	    f.close()
        if (marble_history.history['loss'][marble_length-1])>0.1:
    	    i -=1
        else:
    	    marble_model.save(os.path.join(kpath,str(i),'unet_marble.hdf5'))
        i += 1
    #--------------------------------------------------------------sawdust--------------------------------
    i = 0
    while i < k_value:
    #for i in range(k_value):
        print('Now working on folder -'+str(i))
        test_num = math.floor(n_img/k_value)
        if i == k_value-1:
            test_num += n_img%k_value
        kpath = 'Kfolder'
        #myGene = trainGenerator(batch_size,os.path.join(kpath,str(i),'train'),'image','labels',os.path.join(kpath,str(i),'train','aug'),data_gen_args,)
        sawdustGene = trainGenerator(batch_size,os.path.join(kpath,str(i),'train'),'image','sawdust',None,data_gen_args)
        sawdust_model = unet()
        sawdust_history = sawdust_model.fit(sawdustGene,steps_per_epoch=steps_per_epoch,epochs=epochs,callbacks=[early_stop])
        sawdust_length = len(sawdust_history.history['loss'])
        testGene = testGenerator(os.path.join(kpath,str(i),'test'),num_image = test_num)
        results = sawdust_model.predict(testGene,test_num,verbose=1)
        saveResult_double(i,False,os.path.join(kpath,str(i),'test'),results)
        loss_result = sawdust_history.history['loss']
        csvheader = 'This is fold '+str(i) + 'for sawdust'
        with open('loss_result.csv', 'a', newline ='') as f:
    	    write = csv.writer(f)
    	    write.writerow(csvheader)
    	    write.writerow(loss_result)
    	    f.close()
        if (sawdust_history.history['loss'][sawdust_length-1])>0.1:
    	    i -=1
        else:
    	    sawdust_model.save(os.path.join(kpath,str(i),'unet_sawdust.hdf5'))
        i += 1
else:
    #kfolderGenerator(k_value,'inputs','Labels')
    while i < k_value:
    #for i in range(k_value):
        print('Now working on folder -'+str(i))
        test_num = math.floor(n_img/k_value)
        if i == k_value-1:
            test_num += n_img%k_value
        kpath = 'Kfolder'
        #myGene = trainGenerator(batch_size,os.path.join(kpath,str(i),'train'),'image','labels',os.path.join(kpath,str(i),'train','aug'),data_gen_args,)
        myGene = trainGenerator(batch_size,os.path.join(kpath,str(i),'train'),'image','labels',None,data_gen_args)
        model = unet()
        #model_checkpoint = ModelCheckpoint(os.path.join(kpath,str(i),'unet_marble_'+str(i)+'_.hdf5'), monitor='loss',verbose=1, save_best_only=True)
        #model.fit(myGene,steps_per_epoch=300,epochs=4,callbacks=[model_checkpoint])
        history = model.fit(myGene,steps_per_epoch=steps_per_epoch,epochs=epochs,callbacks=[early_stop])
        length = len(history.history['loss'])
        testGene = testGenerator(os.path.join(kpath,str(i),'test'),num_image = test_num)
        results = model.predict(testGene,test_num,verbose=1)
        saveResult(i,os.path.join(kpath,str(i),'test'),results)
        loss_result = history.history['loss']
        csvheader = 'This is fold '+str(i) + 'for sawdust-marble'
        with open('loss_result.csv', 'a', newline ='') as f:
    	    write = csv.writer(f)
    	    write.writerow(csvheader)
    	    write.writerow(loss_result)
    	    f.close()
        if (history.history['loss'][length-1])>0.1:
    	    i -=1
        else:
    	    model.save(os.path.join(kpath,str(i),'unet_difference.hdf5'))
        i += 1
end = time.time()
time_consumed=end-start
print('-----------------------------------------')
print('It takes '+str(time_consumed)+' to train '+str(k_value)+ ' folds')
