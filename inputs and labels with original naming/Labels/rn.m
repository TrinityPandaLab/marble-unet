File = dir(pwd);
for i = 1:length(File)
    if File(i).isdir == false
        result = zeros(512,512,3);
        Filename = File(i).name;
        pic = imread(Filename);
        result(:,:,1) = pic(:,:,3);
        result(:,:,2) = pic(:,:,3);
        result(:,:,3) = pic(:,:,3);
        newFilename = extractAfter(Filename,3);
        imwrite(result,newFilename);
    end
end
