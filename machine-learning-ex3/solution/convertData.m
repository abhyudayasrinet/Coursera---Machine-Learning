FileData = load('ex3data1.mat');
csvwrite('ex3data1_x.csv', FileData.X);
csvwrite('ex3data1_y.csv', FileData.y);