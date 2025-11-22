classdef counTRObjectCounter
    %counTRObjectCounter CounTR model to count objects based on exemplars.
    %
    %   counter = counTRObjectCounter(exemplarI,exemplarBboxes) constructs
    %   a CounTR object counter model with exemplar patches of objects to
    %   count obtained from exemplarImage at the exemplar bounding box
    %   locations. The exemplar bounding box locations, exemplarBoxes are
    %   specified as a numShots-by-4 numeric matrix with rows of the form
    %   [x y w h], where numShots is the number of axis-aligned rectangles.
    %   The values x and y specify the upper-left corner of the rectangle,
    %   w specifies the width of the rectangle, which is its length along
    %   the x-axis, h specifies the height of the rectangle, which is its
    %   length along the y-axis.
    %
    %   counter = counTRObjectCounter(exemplarPatches) constructs
    %   a CounTR object counter model with exemplar patches of objects to
    %   count. The exemplar patches are specified as numshots-by-1 cell
    %   array with each cell containing a P-by-Q-by-3 patch. All patches are
    %   internally resized to 64-by-64-by-3 patches and stacked to form a
    %   64-by-64-by-3-by-numShots input to the object counter model.
    %
    %   counTRObjectCounter methods:
    %      counTRObjectCounter - Construct CounTR object counter model
    %      countObjects - Count Objects based on exemplars using the CounTR object counter model
    %      densityMap - Compute density map based on exemplars using the CounTR object counter model
    %
    %   Example 1
    %   ---------
    %   % Construct a CounTR Object Counter model
    %
    % I = imread("coins.png");
    % bboxes = ...
    %      [211 149 52 51
    %        86  59 50 51
    %        31  24 54 49];
    % counter = counTRObjectCounter(I,bboxes);
    %
    %   See also counTRObjectCounter/countObjects, counTRObjectCounter/densityMap, anomalyMapOverlay.

    %   Copyright 2024 The MathWorks, Inc.

    properties (Access=private)
        CounTRParams
    end

    properties(SetAccess=private)
        ExemplarPatches
    end

    properties(SetAccess=private,Hidden)
        AllowWritingDensityMapToDisk = true
    end

    properties(SetAccess=private,Dependent)
        NumShots
    end

    properties(SetAccess=private,Dependent,Hidden)
        ExemplarPatchesCell
    end

    methods
        function obj = counTRObjectCounter(I,bboxes)
            arguments
                I
                bboxes (:,4) = []
            end

            vision.internal.requiresNeuralToolbox(mfilename);

            % Make message catalog available in deployed compiler contexts
            visualinspection.internal.setAdditionalResourceLocation();

            % counTRObjectCounter constructor
            obj.CounTRParams = iTripwireCounTRParams;

            if isempty(bboxes)
                validateExemplarPatches(I);
                patchSize = [64 64];
                exemplarPatches = [];
                for pIdx = 1:size(I,1)
                    exemplarPatch = imresize(I{pIdx},patchSize);
                    exemplarPatches = cat(4,exemplarPatches,exemplarPatch);
                end
                obj.ExemplarPatches = exemplarPatches;
            else
                validateExemplarImage(I);
                exemplarImage = I;

                [origHeight,origWidth,~] = size(exemplarImage);

                validateExemplarBBoxes(bboxes);
                % Filter out any invalid box and clip the box to image bounds
                exemplarBBoxes = filterInvalidBBox(bboxes, [origHeight,origWidth]);

                % At least one valid bounding box needs to be provided
                if isempty(exemplarBBoxes)
                    error(message('visualinspection:counTRObjectCounter:insufficientExemplarBBoxes'));
                end

                exemplarImage = resizeAndNormalize(exemplarImage);
                obj.ExemplarPatches = computeExemplarPatches(exemplarImage, exemplarBBoxes, origHeight, origWidth);
            end
        end

        function count = countObjects(obj,X,NameValueArgs)
            % countObjects Count Objects based on exemplars using the CounTR model
            %
            %   count = countObjects(counter,I) counts the number of
            %   objects in the image, I based on the exemplar patches of
            %   objects.
            %
            %   count = countObjects(counter,ds) counts the number of
            %   objects in each image in the datastore, ds based on the
            %   exemplar patches of objects. The same exemplar patches will
            %   be used to find the count in all the images in the
            %   datastore. The output, count will be a vector of size
            %   R-by-1 where R is the number of images in the input
            %   datastore.
            %
            %   [____] = countObjects(___,Name=Value) computes the count with the
            %   following optional Name=Value pairs:
            %
            %       "MiniBatchSize"         - The size of the mini-batches for
            %                                 computing count for datastore input.
            %
            %                                 Default: 128
            %
            %       "ExecutionEnvironment"  - The execution environment for the
            %                                 network. This determines what
            %                                 hardware resources will be used
            %                                 to run the network
            %                                   - "auto" - Use a GPU if it is
            %                                     available, otherwise use CPU.
            %                                   - "gpu" - Use the GPU.
            %                                   - "cpu" - Use the CPU.
            %
            %                                 Default: "auto".
            %
            %       "Verbose"                - Set true to display progress information.
            %
            %                                  Default: true
            %
            % Example: Count objects in an image
            % ----------------------------------
            %
            % I = imread("coins.png");
            % bboxes = ...
            %      [211 149 52 51
            %        86  59 50 51
            %        31  24 54 49];
            % counter = counTRObjectCounter(I,bboxes);
            % count = countObjects(counter,I);

            arguments
                obj counTRObjectCounter
                X {mustBeAValidInferenceInput,mustBeNonempty}
                NameValueArgs.MiniBatchSize {mustBeScalarOrEmpty,mustBeInteger,mustBePositive,mustBeReal} = 128
                NameValueArgs.ExecutionEnvironment string {mustBeMember(NameValueArgs.ExecutionEnvironment,["auto","gpu","cpu"])} = "auto"
                NameValueArgs.Verbose (1,1) {mustBeNumericOrLogical} = true
            end

            % Validate inputs
            executionEnvironment = NameValueArgs.ExecutionEnvironment;

            if isdatastore(X)
                ds = copy(X);
                reset(ds);

                obj.AllowWritingDensityMapToDisk = false;
                NameValueArgs.NameSuffixSource = "filename";
                count = iProcessImageDatastoreSerially(ds, obj, NameValueArgs);
            else
                count = countObjectsInOneImageOrBatch(obj,X,executionEnvironment);
            end
        end

        function dsDensityMap = densityMap(obj,X,NameValueArgs)
            % densityMap - Compute density map using the CounTR model
            %
            %   densityMap = densityMap(counter,I) computes the density map
            %   of the image, I based on the exemplar patches of objects.
            %
            %   dsDensityMap = densityMap(counter,ds) computes the density
            %   map of each image in the datastore, ds based on the
            %   exemplar patches of objects. The same exemplar patches will
            %   be used to find the density map of all the images in the
            %   datastore. The output, dsDensityMap will be a datastore
            %   containing R density maps of size P-by-Q where R is the
            %   number of images in the input datastore and P-by-Q
            %   corresponds to the spatial size of the images in the input
            %   datastore.
            %
            %   [____] = densityMap(___,Name=Value) computes the density map with the
            %   following optional Name=Value pairs:
            %
            %       "MiniBatchSize"         - The size of the mini-batches for
            %                                 computing density maps for datastore input.
            %
            %                                 Default: 128
            %
            %       "ExecutionEnvironment"  - The execution environment for the
            %                                 network. This determines what
            %                                 hardware resources will be used
            %                                 to run the network
            %                                   - "auto" - Use a GPU if it is
            %                                     available, otherwise use CPU.
            %                                   - "gpu" - Use the GPU.
            %                                   - "cpu" - Use the CPU.
            %
            %                                 Default: "auto".
            %
            %       "Verbose"                - Set true to display progress information.
            %
            %                                  Default: true
            %
            %       "WriteLocation"          - Folder location specified as a string scalar, or a character vector. The specified folder must exist and have write permissions.
            %
            %                                  Default: pwd
            %
            %       "NamePrefix"             - Prefix applied to output filenames specified as a string scalar or character vector.
            %
            %                                  Default: "densityMap"
            %
            %       "OutputFolderName"       - Output folder name for density map results, specified as a string scalar or a character vector. This folder is in the location specified by the value of the WriteLocation name-value argument. If the output folder already exists, the function creates a new folder with the string "_1" appended to the end of the name. Set OutputFoldername to "" to write all the results to the folder specified by WriteLocation.
            %
            %                                  Default: "densityMapOutput"
            %
            %       "NameSuffix"             - Suffix to add to the output image filename, specified as a string scalar or a character vector. If you do not specify the suffix, the function uses the input filenames as the output file suffixes. The function extracts the input filenames from the info output of the read object function of the datastore. When the datastore does not provide the filename, the function does not add a suffix.
            %
            %                                  Default: ""
            %
            % Example: Density map and count in an image
            % ------------------------------------------
            %
            % I = imread("coins.png");
            % bboxes = ...
            %      [211 149 52 51
            %        86  59 50 51
            %        31  24 54 49];
            % counter = counTRObjectCounter(I,bboxes);
            % count = countObjects(counter,I);
            % densityMapI = densityMap(counter,I);
            % densityOverlayImage = anomalyMapOverlay(I,densityMapI,"Blend","equal");
            % fontSize = min(floor(size(densityOverlayImage,2)/30),200);
            % gutterWidth = fontSize*9;
            % densityOverlayImageWText = insertText(densityOverlayImage,[size(densityOverlayImage,2)-gutterWidth 5],...
            % sprintf("count=%0.2f",count),FontSize=fontSize);
            % imshow(densityOverlayImageWText)

            arguments
                obj counTRObjectCounter
                X {mustBeAValidInferenceInput,mustBeNonempty}
                NameValueArgs.MiniBatchSize {mustBeScalarOrEmpty,mustBeInteger,mustBePositive,mustBeReal} = 128
                NameValueArgs.ExecutionEnvironment string {mustBeMember(NameValueArgs.ExecutionEnvironment,["auto","gpu","cpu"])} = "auto"
                NameValueArgs.Verbose (1,1) {mustBeNumericOrLogical} = true
                NameValueArgs.WriteLocation (1,1) string = string(pwd)
                NameValueArgs.OutputFolderName (1,1) string = "densityMapOutput"
                NameValueArgs.NamePrefix (1,1) string {iCheckNamePrefix} = "densityMap"
                NameValueArgs.NameSuffix (1,1) string {iCheckNameSuffix} = ""
                NameValueArgs.UseParallel (1,1) {mustBeNumericOrLogical} = false
            end

            % Validate inputs
            executionEnvironment = NameValueArgs.ExecutionEnvironment;

            if ~isdatastore(X) && ...
                    (NameValueArgs.WriteLocation ~= string(pwd) || ...
                    NameValueArgs.NamePrefix ~= "densityMap" ||...
                    NameValueArgs.Verbose ~= true || ...
                    NameValueArgs.UseParallel ~= false)
                warning(message('visualinspection:counTRObjectCounter:onlyApplyWithImdsInput'))
            end

            if isdatastore(X)
                % Only check write location when input is a datastore.
                iCheckWriteLocation(NameValueArgs.WriteLocation);

                iCheckOutputFolderName(NameValueArgs.OutputFolderName);

                if NameValueArgs.NameSuffix == ""
                    % By default, use input file names as the suffix.
                    NameValueArgs.NameSuffixSource = "filename";
                else
                    NameValueArgs.NameSuffixSource = "user";
                end

                ds = copy(X);
                reset(ds);

                NameValueArgs.OutputFolderName = iCreateOutputFolder(NameValueArgs.WriteLocation, NameValueArgs.OutputFolderName);

                if NameValueArgs.UseParallel && obj.AllowWritingDensityMapToDisk
                    [~, filenames] = iProcessImageDatastoreInParallel(ds, obj, NameValueArgs);
                else
                    [~, filenames] = iProcessImageDatastoreSerially(ds, obj, NameValueArgs);
                end

                % Create output imageDatastore
                dsDensityMap = imageDatastore(filenames);
            else
                [~, dsDensityMap] = countObjectsInOneImageOrBatch(obj,X,executionEnvironment);
            end
        end

        function numShots = get.NumShots(obj)
            numShots = size(obj.ExemplarPatches,4);
        end

        function exemplarPatchesCell = get.ExemplarPatchesCell(obj)
            numShots = obj.NumShots;
            exemplarPatchesCell = cell(numShots,1);
            for idx = 1:numShots
                exemplarPatchesCell{idx,1} = obj.ExemplarPatches(:,:,:,idx);
            end
        end
    end

    methods(Access=private)

        function [countForOneImageOrBatch, densityMapForOneImageOrBatch] = countObjectsInOneImageOrBatch(obj,img,executionEnvironment)
            [h,w,~,~] = size(img);
            imgResized = resizeAndNormalize(img);
            [imgResized,castedToDlarray,castedToGpuArray] = iCastInputForInference(imgResized,executionEnvironment);
            [countForOneImageOrBatch, densityMapForOneImageOrBatch] = countObjectsInternal(obj,imgResized);
            densityMapForOneImageOrBatch = imresize(densityMapForOneImageOrBatch,[h,w],"bilinear");
            countForOneImageOrBatch = iUndoCastingForInference(countForOneImageOrBatch,castedToDlarray,castedToGpuArray);
            densityMapForOneImageOrBatch = iUndoCastingForInference(densityMapForOneImageOrBatch,castedToDlarray,castedToGpuArray);
        end

        function [count, densityMap] = countObjectsInternal(obj,img)
            [newHeight, newWidth, ~, newBatch] = size(img);
            densityMap = zeros(newHeight,newWidth,1,newBatch,"like",img);
            start = 1;
            prev = 0;
            if newHeight <= newWidth
                % landscape orientation
                while (start + 383) <= newWidth
                    output = countObjectsInOneWindow(obj,img(:,start:start+383,:,:));
                    output = squeeze(permute(output,[2 3 1]));
                    d1 = padarray(output(:, 1:prev-start+1,:),[0,start-1],0,"pre");
                    d1 = padarray(d1,[0,newWidth-prev],0,"post");

                    d2 = padarray(output(:, prev-start+2:384,:),[0,prev],0,"pre");
                    d2 = padarray(d2,[0, newWidth-start-384+1],0,"post");

                    densityMapLeft = padarray(densityMap(:, 1:start-1,:),[0, newWidth-start+1],0,"post");
                    densityMapMid = padarray(densityMap(:, start:prev,:),[0,start-1],0,"pre");
                    densityMapMid = padarray(densityMapMid, [0,newWidth-prev],0,"post");
                    densityMapRight = padarray(densityMap(:, prev+1:newWidth,:),[0,prev],0,"pre");

                    densityMap = densityMapLeft + densityMapRight + densityMapMid / 2 + d1 / 2 + d2;

                    prev = start + 383;
                    start = start + 128;
                    if start + 383 >= newWidth
                        if start == newWidth - 384 + 128 + 1
                            break
                        else
                            start = newWidth - 384 + 1;
                        end
                    end
                end
            else % portrait orientation
                while (start + 383) <= newHeight
                    output = countObjectsInOneWindow(obj,img(start:start+383,:,:,:));
                    output = squeeze(permute(output,[2 3 1]));
                    d1 = padarray(output(1:prev-start+1,:,:),[start-1,0],0,"pre");
                    d1 = padarray(d1,[newHeight-prev,0],0,"post");

                    d2 = padarray(output(prev-start+2:384,:,:),[prev,0],0,"pre");
                    d2 = padarray(d2,[newHeight-start-384+1,0],0,"post");

                    densityMapLeft = padarray(densityMap(1:start-1,:,:),[newHeight-start+1,0],0,"post");
                    densityMapMid = padarray(densityMap(start:prev,:,:),[start-1,0],0,"pre");
                    densityMapMid = padarray(densityMapMid, [newHeight-prev,0],0,"post");
                    densityMapRight = padarray(densityMap(prev+1:newHeight,:,:),[prev,0],0,"pre");

                    densityMap = densityMapLeft + densityMapRight + densityMapMid / 2 + d1 / 2 + d2;

                    prev = start + 383;
                    start = start + 128;
                    if start + 383 >= newHeight
                        if start == newHeight - 384 + 128 + 1
                            break
                        else
                            start = newHeight - 384 + 1;
                        end
                    end
                end
            end
            count = squeeze(sum(densityMap,[1 2])./60);
            densityMap = reshape(densityMap,[size(densityMap,[1 2]),1,size(densityMap,3)]);
        end

        function output = countObjectsInOneWindow(obj,img)

            exemplarPatches = permute(obj.ExemplarPatches,[5 4 3 1 2]);
            numShots = obj.NumShots;
            output = visualinspection.internal.counTRModelFunction(img,exemplarPatches,numShots,obj.CounTRParams);

        end

    end

    methods (Hidden)

        function s = saveobj(obj)
            s.ExemplarPatchesCell = obj.ExemplarPatchesCell;
        end

    end

    methods (Static,Hidden)
        function obj = loadobj(s)
            obj = counTRObjectCounter(s.ExemplarPatchesCell);
        end
    end

end

function validateExemplarImage(in)
    tf = (isnumeric(in)||islogical(in))&&...
        ndims(in) < 4 && ...
        (size(in,3)==3||size(in,3)==1) && ...
        isreal(in) && allfinite(in) && ~issparse(in);
    if(~tf)
        error(message('visualinspection:counTRObjectCounter:invalidExemplarImage'));
    end
end

function tf = validateExemplarBBoxes(in)
    tf = (isnumeric(in)&&...
        ismatrix(in) && ...
        size(in,2)==4) && ...
        isreal(in) && ...
        allfinite(in) && ~issparse(in);
    if(~tf)
        error(message('visualinspection:counTRObjectCounter:invalidExemplarBBoxes'));
    end
end

function tf = validateExemplarPatches(in)
    tf = iscell(in);
    if tf
        tfIn = cellfun(@(x)(isnumeric(x)||islogical(x))&&...
            ndims(x)==3 && ...
            (size(x,3)==3) && ...
            isreal(x) && ...
            allfinite(x) && ~issparse(x),in);
        tf = all(tfIn);
    end
    if(~tf)
        error(message('visualinspection:counTRObjectCounter:invalidExemplarPatches'));
    end
end

function bbox = filterInvalidBBox(bbox, imSize)
    % Filter boxes outside the image bounds
    H = imSize(1);
    W = imSize(2);

    % Check if the boxes are completely out of bounds, or is second edge is
    % less than positive edge
    if( (bbox(1,3) < 1)||...
            (bbox(1,4) < 1)||...
            (bbox(1,1) > W)||...
            (bbox(1,2) > H))

        bbox = [];
        return;
    end
    % Clip boxes in image bounds
    bbox(1,1) = max(1, bbox(1,1));
    bbox(1,2) = max(1, bbox(1,2));
    bbox(1,3) = min(W, bbox(1,3));
    bbox(1,4) = min(H, bbox(1,4));

end

function iPrintHeader(printer)
    printer.printMessage('visualinspection:counTRObjectCounter:verboseHeader');
    printer.print('--------------------------------------');
    printer.linebreak();
end

function updateMessage(printer, prevMessage, nextMessage)
    backspace = sprintf(repmat('\b',1,numel(prevMessage))); % figure how much to delete
    printer.print([backspace nextMessage]);
end

function nextMessage = iPrintInitProgress(printer, prevMessage, k)
    nextMessage = getString(message('visualinspection:counTRObjectCounter:verboseProgressTxt',k));
    updateMessage(printer, prevMessage(1:end-1), nextMessage);
end

function nextMessage = iPrintProgress(printer, prevMessage, k)
    nextMessage = getString(message('visualinspection:counTRObjectCounter:verboseProgressTxt',k));
    updateMessage(printer, prevMessage, nextMessage);
end

function counTRParams = iTripwireCounTRParams()
    % Check if support package and all dependencies are installed

    persistent cvtInstalled dltInstalled
    if isempty(cvtInstalled)
        [cvtInstalled, ~] = license("checkout", "video_and_image_blockset");
    end

    if isempty(dltInstalled)
        [dltInstalled, ~] = license("checkout", "neural_network_toolbox");
    end

    fullPath = which("counTRObjectCounter");
    dirPath = fileparts(fullPath);
    filesepIndices = regexp(dirPath,filesep);
    dirPath = dirPath(1:filesepIndices(end));
    matfile = fullfile(dirPath,"pretrained-models","counTR","counTR","counTRParams.mat");
    data = load(matfile);
    counTRParams = data.counTRParams;

    if cvtInstalled && dltInstalled
        return;
    end

    % Construct an appropriate error message indicating which products are
    % missing
    cvtName = "";
    dltName = "";
    mainMsgHoleId3 = "";

    if (~cvtInstalled || ~dltInstalled)
        % If toolboxes are missing, list the names of the
        % missing toolboxes
        if (~cvtInstalled || isempty(ver("vision")))
            cvtName = "Computer Vision Toolbox™";
            cvtName = newline() + "* " + cvtName;
        else
            cvtName = "";
        end

        if (~dltInstalled|| isempty(ver("nnet")))
            dltName = "Deep Learning Toolbox™";
            dltName = newline() + "* " + dltName;
        else
            dltName = "";
        end

        mainMsgHoleId3 = getString( message( "visualinspection:counTRObjectCounter:tbxNotInstalledMsg", ...
            cvtName, dltName ) );
    end

    throwAsCaller( MException( message( "visualinspection:counTRObjectCounter:missingDependencies", ...
        cvtName, dltName, ...
        mainMsgHoleId3) ) );
end

function [x,castedToDlarray,castedToGpuArray] = iCastInputForInference(x,executionEnvironment)

    % Allow integer types to flow through for convenience in
    % inference workflows
    if ~isfloat(x)
        x = single(x);
    end

    castedToDlarray = false;
    castedToGpuArray = false;
    if ~isgpuarray(x) && ((executionEnvironment == "gpu") || (canUseGPU && (executionEnvironment == "auto")))
        x = gpuArray(x);
        castedToGpuArray = true;
    end

end

function x = iUndoCastingForInference(x,castedToDlarray,castedToGpuArray)
    if castedToDlarray
        x = extractdata(x);
    end

    if castedToGpuArray
        x = gather(x);
    end
end

function [imgResized,inputMin,inputMax] = resizeAndNormalize(I)

    [h,w,~] = size(I);

    if h <= w
        newWidth  = 384;
        newHeight = (16 * floor((w / h * 384) / 16));
    else
        newWidth  = (16 * floor((h / w * 384) / 16));
        newHeight = 384;
    end

    % Convert grayscale to 3-channel input
    if(size(I,3)==1)
        I = repmat(I,[1 1 3]);
    end

    imgResized = imresize(I,[newWidth newHeight],"bilinear");
    [imgResized,inputMin,inputMax] = normalizeImage(imgResized);
end

function [imgNorm,inputMin,inputMax] = normalizeImage(img,NameValueArgs)
    arguments
        img
        NameValueArgs.InputMin = min(img,[],"all");
        NameValueArgs.InputMax = max(img,[],"all");
    end
    inputMin = NameValueArgs.InputMin;
    inputMax = NameValueArgs.InputMax;
    imgNorm = rescale(img,InputMin=inputMin,InputMax=inputMax);
end

function TF = isdatastore(x)
    TF = isa(x,"matlab.io.Datastore") || isa(x,"matlab.io.datastore.Datastore");
end

function exemplarPatches = computeExemplarPatches(imgResized, boxes, origHeight, origWidth)
    [newHeight, newWidth,~] = size(imgResized);
    scaleFactorHeight = newHeight / origHeight;
    scaleFactorWidth = newWidth / origWidth;
    patchSize = [64 64];
    exemplarPatches = [];
    bboxes = zeros(size(boxes));
    for b = 1:size(boxes,1)
        bboxes(b,:) = [floor(boxes(b,1)*scaleFactorWidth),...
            floor(boxes(b,2)*scaleFactorHeight),...
            floor(boxes(b,3)*scaleFactorWidth),...
            floor(boxes(b,4)*scaleFactorHeight)];
        exemplarPatch = imcrop(imgResized,bboxes(b,:));
        exemplarPatch = imresize(exemplarPatch,patchSize);
        exemplarPatches = cat(4,exemplarPatches,exemplarPatch);
    end
end

function iCheckWriteLocation(x)
    validateattributes(x, {'char','string'}, {'scalartext'}, ...
        mfilename, 'WriteLocation')

    if ~exist(x,'dir')
        error(message('visualinspection:counTRObjectCounter:dirDoesNotExist'));
    end

    vision.internal.inputValidation.checkWritePermissions(x);
end

function iCheckOutputFolderName(x)
    validateattributes(x, {'char','string'}, {'scalartext'}, ...
        mfilename, 'OutputFolderName')
end

function iCheckNamePrefix(x)
    validateattributes(x, {'char','string'}, {'scalartext'}, ...
        mfilename, 'NamePrefix')
end

function iCheckNameSuffix(x)
    validateattributes(x, {'char','string'}, {'scalartext'}, ...
        mfilename, 'NameSuffix')
end

function filenames = iWriteDensityMapData(densityMap, indices, params, N, info)
    writeLocation = params.WriteLocation;
    name = iCreateFileName(params.NamePrefix, indices, N, info);
    filenames = iPrependOutputLocation(name, writeLocation, params.OutputFolderName);
    iWriteImageBatch(densityMap,filenames);
end

function name = iCreateFileName(prefix, idx, numImages, info)
    % Choose PNG format for 2-D images
    ext = 'png';

    % Determine the output file format. When the number of observations
    % is nonfinite, simply append the file ID. Otherwise, use %0d to have the ID
    % string based on the number of observations. For example, for 1000
    % observations, produce label_0001, label_0002, ..., label_1000.
    if isfinite(numImages)
        format = sprintf('%%s_%%0%dd%%s.%s', string(numImages).strlength,ext);
    else
        format = sprintf('%%s_%%d%%s.%s',ext);
    end

    % Generate the filenames.
    name = cell(numel(idx),1);
    for i = 1:numel(idx)
        suffix = string(info(i));
        if ~isequal(suffix,"")
            [~,suffix,~] = fileparts(suffix);
            suffix = "_" + suffix;
        end
        name{i} = sprintf(format, prefix, idx(i), suffix);
    end
end

function outputFolderName = iCreateOutputFolder(writeLocation, outputFolderName)
    if isequal(strlength(outputFolderName),0)
        % Output foldername is empty, write results into write location.
        outputFolderName = '';
    else
        % Create output folder.
        outputFolderName = iCreateUniqueFoldername(writeLocation, outputFolderName);
        outputLocation = fullfile(writeLocation, outputFolderName);
        try
            [success, msg, msgId] = mkdir(outputLocation);
            if ~success
                throw(MException(msgId,msg));
            end
        catch ME
            iThrowUnableToCreateOutputFolderMessage(ME,outputLocation);
        end
    end
end

function uniqueName = iCreateUniqueFoldername(writeLocation, outputFolderName)
    putativeLocation = fullfile(writeLocation, outputFolderName);
    uniqueName = outputFolderName;
    k = 0;
    while exist(putativeLocation) %#ok<EXIST>
        k = k + 1;
        uniqueName = outputFolderName + "_" + k;
        putativeLocation = fullfile(writeLocation, uniqueName);
    end
    uniqueName = char(uniqueName);
end

function iThrowUnableToCreateOutputFolderMessage(cause,outputFolderName)
    msg = message('visualinspection:counTRObjectCounter:UnableToCreateOutputFolder',outputFolderName);
    exception = MException(msg);
    exception = addCause(exception,cause);
    throwAsCaller(exception);
end

function folder = iPrependOutputLocation(filename, writeLocation, outputFolderName)
    folder = fullfile(writeLocation, outputFolderName, filename);
end

function iWriteImageBatch(I,names)
    for i = 1:numel(names)
        imwrite(I(:,:,:,i),names{i});
    end
end

function [count, filenames] = iProcessImageDatastoreSerially(ds, obj, params)
    numImages = iNumberOfObservations(ds);

    if isfinite(numImages)
        filenames = strings(numImages,1);
    else
        filenames = strings(0,0);
    end

    printer = vision.internal.MessagePrinter.configure(params.Verbose);

    iPrintHeader(printer);
    msg = iPrintInitProgress(printer,'', 1);

    loader = iCreateDataLoader(ds,params);

    % Iterate through data and write results to disk.
    count = single([]);
    k = 1;
    while hasdata(loader)
        out = nextBatch(loader);

        X = out{1};
        info = out{2};

        [batchSize, isDataBatched] = iDetermineBatchSize(X);

        idx = k:k+batchSize-1;

        if isDataBatched
            [countForOneBatch, densityMapForOneBatch] = countObjectsInOneImageOrBatch(obj,X,params.ExecutionEnvironment);
            count = cat(1,count,countForOneBatch);
            if obj.AllowWritingDensityMapToDisk
                filenames(idx) = iWriteDensityMapData(densityMapForOneBatch, idx, params, numImages, info);
            end
            msg = iPrintProgress(printer, msg, idx(end));

        else
            for i = 1:numel(idx)
                [countForOneImage, densityMapForOneImage] = countObjectsInOneImageOrBatch(obj,X{i},params.ExecutionEnvironment);
                count = cat(1,count,countForOneImage);
                if obj.AllowWritingDensityMapToDisk
                    filenames(idx(i)) = iWriteDensityMapData(densityMapForOneImage, idx(i), params, numImages, info(i));
                end
                msg = iPrintProgress(printer, msg, idx(i));

            end
        end
        k = idx(end)+1;
    end
    printer.linebreak(2);
end

function [count, filenames] = iProcessImageDatastoreInParallel(ds, obj, params)

    isLocalPoolOpen = iAssertOpenPoolIsLocal();

    if ~isLocalPoolOpen
        tryToCreateLocalPool();
    end

    numImages = iNumberOfObservations(ds);

    if isfinite(numImages)
        filenames = strings(numImages,1);
    else
        filenames = strings(0,0);
    end

    printer = vision.internal.MessagePrinter.configure(params.Verbose);

    iPrintHeader(printer);

    msg = iPrintInitProgress(printer,'', 1);

    % pre-allocate future buffer.
    futureWriteBuffer = parallel.FevalFuture.empty();

    loader = iCreateDataLoader(ds,params);

    % Iterate through data and write results to disk.
    count = single([]);
    k = 1;
    while hasdata(loader)
        out = nextBatch(loader);

        X = out{1};
        info = out{2};

        [batchSize, isDataBatched] = iDetermineBatchSize(X);

        idx = k:k+batchSize-1;

        if isDataBatched
            [countForOneBatch, densityMapForOneBatch] = countObjectsInOneImageOrBatch(obj,X,params.ExecutionEnvironment);
            count = cat(1,count,countForOneBatch);
            [futureWriteBuffer, filenames(idx)] = ...
                iParallelWriteDensityMapData(densityMapForOneBatch, idx, params, futureWriteBuffer, numImages, info);

            msg = iPrintProgress(printer, msg, idx(end));

        else

            for i = 1:numel(idx)
                [countForOneImage, densityMapForOneImage] = countObjectsInOneImageOrBatch(obj,X{i},params.ExecutionEnvironment);
                count = cat(1,count,countForOneImage);

                [futureWriteBuffer, filenames(idx(i))] = ...
                    iParallelWriteDensityMapData(densityMapForOneImage, idx(i), params, futureWriteBuffer, numImages, info(i));

                msg = iPrintProgress(printer, msg, idx(i));
            end
        end
        k = idx(end)+1;
    end

    % wait for all futures to finish
    fetchOutputs(futureWriteBuffer);
    iErrorIfAnyFutureFailed(futureWriteBuffer);

    printer.linebreak(2);
end

function pool = tryToCreateLocalPool()
    defaultProfile = ...
        parallel.internal.settings.ProfileExpander.getClusterType(parallel.defaultProfile());

    if(defaultProfile == parallel.internal.types.SchedulerType.Local)
        % Create the default pool (ensured local)
        pool = parpool;
    else
        % Default profile not local
        error(message('vision:vision_utils:noLocalPool', parallel.defaultProfile()));
    end
end

function TF = iAssertOpenPoolIsLocal()
    pool = gcp('nocreate');
    if isempty(pool)
        TF = false;
    else
        if pool.Cluster.Type ~= parallel.internal.types.SchedulerType.Local
            error(message('vision:vision_utils:noLocalPool', pool.Cluster.Type));
        else
            TF = true;
        end
    end
end

function [futureWriteBuffer, filename] = ...
        iParallelWriteDensityMapData(densityMap, idx, params, futureWriteBuffer, numImages, info)
    % Push write operation onto future buffer. First remove finished futures.
    % If buffer is full, wait till one complete then pop it from the buffer.
    %
    % densityMap can be a single image or a batch of images. idx is a scalar for a
    % single image and a vector for a batch of images.

    iErrorIfAnyFutureFailed(futureWriteBuffer);

    % Remove finished futures.
    finished = arrayfun(@(f)strcmp(f.State,'finished'),futureWriteBuffer);
    futureWriteBuffer(finished) = [];

    % Add to future buffer.
    filename = iCreateFileName(params.NamePrefix, idx, numImages, info);
    filename = iPrependOutputLocation(filename, params.WriteLocation, params.OutputFolderName);

    futureWriteBuffer(end+1) = parfeval(...
        @iWriteImageBatch, 0, uint8(densityMap), filename);

    if length(futureWriteBuffer) > params.MiniBatchSize
        % Buffer is full. Wait till one of the futures is done.
        idx = fetchNext(futureWriteBuffer);
        futureWriteBuffer(idx) = [];
    end
end

function iErrorIfAnyFutureFailed(futures)
    failed = arrayfun(@(x)strcmpi(x.State,'failed'), futures);

    if any(failed)
        % kill existing work and throw error.
        for i = 1:numel(futures)
            futures(i).cancel();
        end

        throw(futures(find(failed,1)).Error);
    end
end

function out = iTryToBatchData(X)
    try
        channelDims = 4;
        if iscell(X)
            batch = cat(channelDims,X{:,1});
        end
    catch
        % Return X as-is.
        batch = X(:,1);
    end
    out = {batch  X(:,2)};
end

function n = iNumberOfObservations(ds)
    if isa(ds,'matlab.io.datastore.ImageDatastore') || isPartitionable(ds)
        n = numpartitions(ds);
    else
        n = inf;
    end
end

function loader = iCreateDataLoader(ds,params)
    if params.NameSuffixSource == "filename"
        ds = transform(ds,@(data,info)iExtractFilename(data,info),'IncludeInfo',true);
    else
        ds = transform(ds,@(data,info)iAddCustomSuffix(data,info,params.NameSuffix),'IncludeInfo',true);
    end
    loader = nnet.internal.cnn.DataLoader(ds,...
        'MiniBatchSize',params.MiniBatchSize,...
        'CollateFcn',@(x)iTryToBatchData(x));
end

function [data,info] = iExtractFilename(data,info)
    if ~iscell(data)
        data = {data};
    end
    N = size(data,1);
    if isfield(info,'Filename')
        try
            data = [data info.Filename];
        catch
            % Number of data and filename is not 1-to-1. Unable to add filename
            % as a suffix.
            data = [data repelem({''},N,1)];
        end
    else
        data = [data repelem({''},N,1)];
    end
end

function [data, info] = iAddCustomSuffix(data,info,suffix)
    if ~iscell(data)
        data = {data};
    end
    N = size(data,1);
    data = [data repelem({suffix},N,1)];
end

function [batchSize, isDataBatched] = iDetermineBatchSize(X)
    if iscell(X)
        batchSize = numel(X);
        isDataBatched = false;
    else
        obsDim = 4;
        batchSize = size(X,obsDim);
        isDataBatched = true;
    end
end

function mustBeAValidInferenceInput(x)
    if (~isnumeric(x) || isdlarray(x)) && ~isdatastore(x)
        error(message('visualinspection:counTRObjectCounter:invalidInferenceInput'));
    end

    % Includes gpuArray
    if isnumeric(x)
        if ndims(x) > 3
            error(message('visualinspection:visualInspectionGeneral:invalidInputDimensionality',3));
        end
        validateattributes(x, {'logical' 'numeric'}, {'real', 'nonsparse', 'finite'}, ...
            mfilename, 'I', 2);
    end
end

