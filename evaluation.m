 %evaluation of IOU
 addpath('/home/zyck/Downloads/unet-test/unet-master/Kfolder/total/ans');
 addpath('/home/zyck/Downloads/unet-test/unet-master/Kfolder/total/labels');
 myDir = '/home/zyck/Downloads/unet-test/unet-master/Kfolder/total/ans'; %gets directory
 myFiles = dir(fullfile(myDir,'*.tif')); %gets all tif files in struct
 result_IOU= [];
 result_Dice= [];
 for k = 1:length(myFiles)
   baseFileName = myFiles(k).name;
   iname=extractBefore(baseFileName,'.tif');
   a=imread(strcat(iname,'_predict.png'));
   b=imread(baseFileName);
   b=b(:,:,1);
   b=imresize(b,[256,256]);
   a=imbinarize(a,0.5);
   b=imbinarize(b,0.5);
   intersection=((a==b)&(a==0));
   a=-1*a+1;
   b=-1*b+1;
   count=sum(intersection(intersection==1));
   union=sum(a(a==1))+sum(b(b==1))-count;
   sumTwo = sum(a(a==1)) + sum(b(b==1));
   Dice_value = 2*count/sumTwo;
   IOU_value = count/union;
   result_IOU = [result_IOU,IOU_value];
   result_Dice = [result_Dice,Dice_value];
 end
 %writematrix(result_IOU,'/home/zyck/Downloads/unet-test/unet-master/result.xlsx','Sheet',1,'Range','I21')
 %writematrix(result_Dice,'/home/zyck/Downloads/unet-test/unet-master/result.xlsx','Sheet',1,'Range','I58')