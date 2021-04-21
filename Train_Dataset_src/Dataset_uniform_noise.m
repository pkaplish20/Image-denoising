%created by Rupali
%Dataset link : http://places2.csail.mit.edu/download.html
clc;
close all;
%save the path of the input and output images 
inputPath = 'D:\dataset\colorUniform\colorUniformCleanImage';
destinationPath = 'D:\dataset\colorVarying\cleanVaryingNoisyImage';
%read all the images of format jpg
myJpgs = dir(fullfile(inputPath,'*.jpg'));
disp(length(myJpgs));
%make a new directory to save the filtered image
 mkdir(destinationPath);
 for d = 1:length(myJpgs)
     srcFileName      = fullfile(inputPath, myJpgs(d).name);
     I = imread(srcFileName);
     G = imnoise(I,'gaussian',0.02); 
     int_vals = mat2gray(G);
     [~,filename,ext] = fileparts(srcFileName);
     newFileName      = sprintf('%s_noise%s',filename,ext);
     destFileName     = fullfile(destinationPath,newFileName);
     imwrite(int_vals, destFileName);
     disp(myJpgs(d).name)
 end
 