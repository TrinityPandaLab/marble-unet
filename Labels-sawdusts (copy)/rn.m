for k = 0:525
    i = imread(strcat(string(k)+'.bmp'));
    i = i(:,:,1);
    h = zeros([640,640]);
    h = 1-h;

    h(1:height(i),1:length(i)) = i;
    h = imresize(h,[512,512],'nearest');
    imwrite(h,strcat(string(k)+'.tif'));
end
