%evaluation within one folder
 %evaluation of IOU
 prediction_path = 'prediction/';
 ans_path = 'ans/';
 files = dir(strcat(ans_path,'*.tif'));
 result_IOU= [];
 for img = files'
   ansname = strcat(ans_path,img.name);
   prename = strcat(prediction_path,string(extractBetween(img.name,'sawdust','.tif')),'_predict.png');
   a=imread(ansname);
   b=imread(prename);
   a=a(:,:,1);
   b=b(:,:,1);
   a=imresize(a,[256,256]);
   b=imresize(b,[256,256]);
   a=imbinarize(a,0.5);
   b=imbinarize(b,0.5);
   intersection=((a==b)&(a==0));
   a=-1*a+1;
   b=-1*b+1;
   count=sum(intersection(intersection==1));
   union=sum(a(a==1))+sum(b(b==1))-count;
   IOU_value = count/union;
   result_IOU = [result_IOU,IOU_value];
 end
 writematrix(result_IOU,'/home/zyck/Downloads/unet-test/unet-master/result.xlsx','Sheet',1,'Range','I47')