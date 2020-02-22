fp = fopen('train-images-idx3-ubyte', 'rb');

fread(fp, 1, 'int32', 0, 'ieee-be');
numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
numCols = fread(fp, 1, 'int32', 0, 'ieee-be');

images = fread(fp, inf, 'unsigned char');
images = reshape(images, numCols * numRows, numImages);
% images = images / 255;
fclose(fp);

fp = fopen('train-labels-idx1-ubyte', 'rb');

metadata = fread(fp, 1, 'int32', 0, 'ieee-be');
numImages = fread(fp, 1, 'int32', 0, 'ieee-be');

labels = fread(fp, inf, 'unsigned char')' / 10;

fp = fopen('t10k-images-idx3-ubyte', 'rb');

fread(fp, 1, 'int32', 0, 'ieee-be');
numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
numCols = fread(fp, 1, 'int32', 0, 'ieee-be');

testImages = fread(fp, inf, 'unsigned char');
testImages = reshape(testImages, numCols * numRows, numImages);

% testImages = testImages / 255;
fclose(fp);

fp = fopen('t10k-labels-idx1-ubyte', 'rb');

metadata = fread(fp, 1, 'int32', 0, 'ieee-be');
numImages = fread(fp, 1, 'int32', 0, 'ieee-be');

testLabels = fread(fp, inf, 'unsigned char')' / 10;