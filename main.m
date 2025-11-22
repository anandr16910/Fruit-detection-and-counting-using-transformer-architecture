clear all;
close all;

%dataDir = fullfile(tempdir,"FruitDetectionDataset");
dataDir = fullfile(pwd, "FruitDetectionDataset");


% User input for the desired class
% User input for the desired class
% User input for the desired fruit
% Create a figure for user input
hFig = figure('Name', 'Fruit Search', 'NumberTitle', 'off', 'MenuBar', 'none', 'Position', [100, 100, 500, 500]);

% Create a text box for user input
uicontrol('Style', 'text', 'String', 'Enter a pattern to search for (e.g., "blueberry", "kiwis", "apple", "cherry","strawberry"):', ...
          'Position', [20 390 400 100], 'FontSize', 16, 'BackgroundColor', 'white', 'ForegroundColor', [0.4 0.2 0]); % Dark brown color

% Set the background image for the figure
axes('Position', [0 0 1 1]); % Create an axes that covers the entire figure
imshow('imgpattern.png'); % Display the background image
uistack(gca, 'bottom'); % Send the axes to the back

inputBox = uicontrol('Style', 'edit', 'Position', [20 50 400 30],'FontSize',16);

% Create a button to submit the input
uicontrol('Style', 'pushbutton', 'String', 'Search', 'Position', [20 10 100 30], ...
          'Callback', @(src, event) searchImages(inputBox, dataDir));

% Wait for the user to close the figure
%uiwait(hFig);

function [filteredFileList, matchedFiles] = searchImages(inputBox, dataDir)
    userInputPattern = get(inputBox, 'String');
    pattern = strcat("\w*", userInputPattern, "\w*");
    
    fileList = dir(fullfile(dataDir, "**"));
    folderNames = {fileList.folder};
    fileNames = {fileList.name};

    matchedFiles = cellfun(@(x) ~isempty(regexpi(x, pattern, "once")) && ...
        isempty(regexpi(x, "json", "once")), fileNames);

    filteredFileList = cellfun(@(x,y) string(x) + filesep + string(y), ...
        folderNames(matchedFiles), fileNames(matchedFiles), 'UniformOutput', false);

    % Display the number of found images in a message box
    msgbox(sprintf("Found %d images containing %s.", numel(filteredFileList), userInputPattern), 'Search Result');
    drawnow;
    maxImages1 = min(15, numel(filteredFileList));
    
 imgCell = cell(1, maxImages1);   

for i = 1:maxImages1
    img = imread(filteredFileList{i});
    imgCell{i} = imresize(img, [500 500]);
end

% Choose a grid: e.g., display as 3 rows x 5 columns if possible
rows = ceil(maxImages1 / 5);
cols = min(5, maxImages1);  % max 5 columns
% Create a new figure for the montage display
f = figure('Name', 'Image Montage', 'NumberTitle', 'off','Visible','on');

% Display the montage of images
montage(imgCell, 'Size', [rows cols]);
title(sprintf('Montage of up to %d images containing ''%s'' in the filename', maxImages1, userInputPattern));


filteredFileList = arrayfun(@(x,y) string(x) + filesep + string(y), ...
    folderNames(matchedFiles), fileNames(matchedFiles), 'UniformOutput', false);
exemplarImage = filteredFileList{1};
I = imread(exemplarImage);
I = imresize(I, [512 512]);

f = figure('Name', 'Boundingboxes', 'NumberTitle', 'off','Visible','on');

exemplarBboxes = selectBoundingBoxes(I);
annotatedImage = insertShape(I,"rectangle",exemplarBboxes,"LineWidth",3);
imshow(annotatedImage)
title("Original Image and Selected Exemplars")

delete(gcp('nocreate'));
parpool('Processes');
%printf(" --- Count operation for selected exemplar")

counTRObj = counTRObjectCounter(I,exemplarBboxes);
%counTRObj.AllowWritingDensityMapToDisk = false;

tic
count = countObjects(counTRObj,I);
msgbox(sprintf("Number of objects detected: %0.3f",count),'Number of fruits detected');
toc
f = figure('Name', 'DensityMap', 'NumberTitle', 'off','Visible','on');
density_map = densityMap(counTRObj,I);
imshow(density_map)

densityOverlayImage = anomalyMapOverlay(I,density_map,"Blend","equal");
fontSize = min(floor(size(densityOverlayImage,2)/30),200);
gutterWidth = fontSize*9;
densityOverlayImageWText = insertText(densityOverlayImage,[size(densityOverlayImage,2)-gutterWidth 5], ...
    sprintf("count=%0.2f",count),FontSize=fontSize);
f = figure('Name', 'DensityOverlay', 'NumberTitle', 'off','Visible','on');
imshow(densityOverlayImageWText);

annotatedInputImage = insertShape((I),"rectangle",exemplarBboxes,LineWidth=10,ShapeColor="red");
f = figure('Name', 'DensityOverlayannotated', 'NumberTitle', 'off','Visible','on');
montage({(I),densityOverlayImageWText,rescale(density_map)},Size=[1 3],BorderSize=[0 10])
%printf(" ----   ")
filteredFileList = cellstr(filteredFileList);
dsFilteredImages = imageDatastore(filteredFileList,FileExtensions=[".jpg",".png"]);
%printf(" --- Count operation for dsFiltered  --- ")

%counts = countObjects(counTRObj, dsFilteredImages);
% Start parallel computing


% Perform the count operation for the filtered images
counts = countObjects(counTRObj, dsFilteredImages);



f = figure('Name', 'Filtered', 'NumberTitle', 'off','Visible','on');

scatter(1:numel(dsFilteredImages.Files),counts)

dsDensityMaps = densityMap(counTRObj,dsFilteredImages);

numDensityMaps = numpartitions(dsDensityMaps);

userResponse = msgbox('Do you want to proceed with further parts of the code?', ...
    'Continue?', 'modal');

waitfor(userResponse); % Wait for the user to close the message box
userResponse = questdlg('Do you want to proceed with further parts of the code?', ...
    'Continue?', 'Yes', 'No', 'Yes');

if strcmp(userResponse, 'No')
    return; % Exit if the user chooses not to proceed
end
%imgIdx =23;

f = figure('Name', 'Image_Idx', 'NumberTitle', 'off','Visible','on');

imgIdx = 2;
maxImages = numel(filteredFileList);
% Create a slider to navigate through the images
hSlider = uicontrol('Style', 'slider', 'Min', 1, 'Max', maxImages, ...
    'Value', imgIdx, 'Position', [100, 50, 300, 20], ...
    'Callback', @(src, event) updateImage(round(get(src, 'Value'))));

% Terminate the parallel pool
delete(gcp('nocreate'));

% Create a text label to display the current image index
hText = uicontrol('Style', 'text', 'Position', [100, 80, 300, 20], ...
    'String', sprintf('Image %d of %d', imgIdx, maxImages));

% Function to update the displayed image based on the slider value
function updateImage(index)
    imgIdx = index;
    x = read(subset(dsImageAndDensityMap, imgIdx));
    I = x{1};
    density_map = x{2};
    count = counts(imgIdx);
    densityOverlayImageWText = overlayDensityMap(I, density_map, count);
    imshow(densityOverlayImageWText);
    set(hText, 'String', sprintf('Image %d of %d', imgIdx, maxImages));
end
dsImageAndDensityMap = combine(dsFilteredImages,dsDensityMaps);
x = read(subset(dsImageAndDensityMap,imgIdx));
I = x{1};
density_map = x{2};
count = counts(imgIdx);
densityOverlayImageWText = overlayDensityMap(I,density_map,count);
f = figure('Name', 'imageIdx', 'NumberTitle', 'off','Visible','on');
imshow(densityOverlayImageWText)
pause(120);
userResponse = questdlg('Do you want to exit the app?', ...
    'Continue?', 'Yes', 'No', 'Yes');

if strcmp(userResponse, 'Yes')
    close all;
    return; % Exit if the user chooses not to proceed
end
end
