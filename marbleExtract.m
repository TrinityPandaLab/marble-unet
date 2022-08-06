%marble-sawdustextract
files = dir('*.tif');
for file = files'
    image = imread(file.name);
    image = imbinarize(image,0.5);
    image = image(:,:,1);
    image(383,:) = 255;
    marble =image;
    CC = bwconncomp(image);
    numPixels = cellfun(@numel,CC.PixelIdxList);
    [biggest,idx] = max(numPixels);
    marble(CC.PixelIdxList{idx}) = 0;
    
    compensate = marble;
    marble = ~marble;
    contour = image - compensate;
    imshow(contour);
    imwrite(marble,strcat('marble',file.name));
    imwrite(contour,strcat('sawdust',file.name));
end
