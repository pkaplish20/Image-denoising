%created by Rupali
%Dataset link : http://places2.csail.mit.edu/download.html
clc;
close all;
%save the path of the input and output images 
inputPath = 'D:\dataset\colorThreeNoise\colorTVaryingNoiseCleanImage';
destinationPath = 'D:\dataset\colorThreeNoise\colorVaringNoisyImage';
%read all the images of format jpg
myJpgs = dir(fullfile(inputPath,'*.jpg'));
%make a new directory to save the filtered image
 mkdir(destinationPath);
 for d = 1:length(myJpgs)
     srcFileName      = fullfile(inputPath, myJpgs(d).name);
     I = imread(srcFileName);
     endCounter = randi([1 3],1,1);
     for  a = 1:endCounter
         if a == 1
             r = randi([1 3],1,1)/100;
             G = imnoise(I,'gaussian',r); 
         else
             r = randi([1 3],1,1)/100;
             G = imnoise(G,'gaussian',r);
         end
     end
     int_vals = mat2gray(G);
     [~,filename,ext] = fileparts(srcFileName);
     newFileName      = sprintf('%s_noise%s',filename,ext);
     destFileName     = fullfile(destinationPath,newFileName);
     imwrite(int_vals, destFileName);
     disp(myJpgs(d).name)
 end