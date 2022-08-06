%convert and combine
sawdust_path = 'Kfolder/total/sawdust/';
marble_path = 'Kfolder/total/marble/';
out_path = 'Kfolder/total/labels/';
k_value = 10;
total_num = 526;
num_per_fold = floor(total_num/10);
remainder = mod(total_num,k_value)+num_per_fold;
for k = 0:1:(k_value-1)
    round = num_per_fold-1;
    if k == k_value-1
        round = remainder-1;
    end
    for index = 0:1:round
        filename = strcat(string(k),'_',string(index),'_','predict.png');
        marble = imread(strcat(marble_path,filename));
        sawdust = imread(strcat(sawdust_path,filename));
        marble = imbinarize(marble,0.5);
        sawdust = imbinarize(sawdust,0.5);
        temp = 1-marble;
        result = temp+sawdust;
        imwrite(result,strcat(out_path,filename));
    end
end



