clear all
close all


% Add Gabor toolbox here
%addpath 'C:\Users\hp\Dropbox\Research\Toolboxes\gabor'

stg=4;
ori=4;

gaborBank = filter_bank(stg, ori, sz_imgs);
Data1 = feature_extraction('Group1', stg, ori, gaborBank); % Class 1 images
Data2 = feature_extraction('Group2', stg, ori, gaborBank); % Class 2 images

save('Data1.mat', 'Data1')
save('Data2.mat', 'Data2')



function [fea] = feature_extraction(folder, stg, ori, gaborBank)
    load OsteoNet.mat   % pretrained CNN
    d = dir(strcat('Base_Folder', folder)); % path where files are lying
    tra_data_sz = round(size(d,1)*0.2);
    i=1;
    cnt = 0;
    fea = zeros(size(d,1)-2, 1024);

    for j=1: size(d,1)
       if (~(strcmp(d(j,1).name,'.') || (strcmp(d(j,1).name,'..'))))
           name = strcat('Base_Folder', folder,'\',d(j,1).name);
           
    
            orig_img = double(imread(name));
            orig_img = orig_img./max(orig_img(:));
            orig_img = imnoise(orig_img,'gaussian',0, 0.01568);
    
            fprintf('Processing image: %d\n', j);
    
            img_mu = mean(orig_img(:));
            img_var = (std(orig_img(:)));
            orig_img = (orig_img-img_mu)./img_var; % uncommented
            fResp = sg_filterwithbank(orig_img,gaborBank);
            fResp2 = sg_filterwithbank2(orig_img,gaborBank);
            fResp = sg_resp2samplematrix(fResp);
            fResp = sg_normalizesamplematrix(fResp);
    

            for l=1:size(fResp,3)
                temp = fResp(:,:,l);
                temp = temp(:);
                resp(:,l) = temp;
            end
            
            fea_rows = zeros(160000,ori);
            fea_cols = zeros(160000,stg);
            eig_temp = zeros(160000, 4);
            for l=1:160000
                temp = abs(resp(l,:));
                temp = temp(:);
                temp = reshape(temp-mean(temp(:)), ori, stg);           
                
                eig_temp(l, :) = abs(eig(temp));
            end
    
            temp = reshape(sum(eig_temp,2), 400, 400);
           temp = activations(net, temp, 'fc_1');
           fea(j, :) = temp(:);
       end
    end
end


function [img] = addnoise(img, snr)

const_noise = 1/(10^(snr/10));
v = var(img(:)) / const_noise;
img = imnoise(img, 'gaussian', 0, v);
end

function [gaborBank] = filter_bank(stg, ori, sz_imgs)

    max_freq = 0.5;
    gaborBank = sg_createfilterbank(sz_imgs, max_freq, stg, ori, 'p', 0.5, 'k', 2.2, 'verbose', 0);
    Data = zeros(5612,128);
    
    
    for j=1:stg
        for k=1:ori
            f2 = ifftshift(ifft2((gaborBank.freq{j}.orient{k}.filter)));
            f2 = hilbert(hilbert(f2)')';
            gaborBank.freq{j}.orient{k}.filter = abs((fft2(f2)));
        end
    end
end


function [net] = Designed_CNN (training_data, training_labels)

    % Set up the layers of the CNN
    layers = [
        imageInputLayer([400 400 1])
        convolution2dLayer(3,32,'Padding','same')
        reluLayer
        maxPooling2dLayer(2,'Stride',2)
        convolution2dLayer(3,64,'Padding','same')
        reluLayer
        maxPooling2dLayer(2,'Stride',2)
        convolution2dLayer(3,128,'Padding','same')
        reluLayer
        maxPooling2dLayer(2,'Stride',2)
        fullyConnectedLayer(1024)
        reluLayer
        fullyConnectedLayer(512)
        reluLayer
        fullyConnectedLayer(1)
        sigmoidLayer
        classificationLayer];
    
    % Set up the training options
    options = trainingOptions('sgdm', ...
        'MaxEpochs',10, ...
        'MiniBatchSize',32, ...
        'Shuffle','every-epoch', ...
        'Verbose',false, ...
        'Plots','training-progress');
    
    training_labels = categorical(training_labels);
    lgraph = layerGraph(layers);

    % Train the CNN
    net = trainNetwork(training_data,training_labels,lgraph,options);
end

