function WriteToFile(fname, submittal)
    cHeader = {'Id', 'label'}; %dummy header
    commaHeader = [cHeader;repmat({','},1,numel(cHeader))]; %insert commaas
    commaHeader = commaHeader(:)';
    textHeader = cell2mat(commaHeader); %cHeader in text with commas
    textHeader = textHeader(1:end-1);
    %write header to file
    fid = fopen(['./Submittals/' fname],'w'); 
    fprintf(fid,'%s\n',textHeader);
    fclose(fid);
    %write data to end of file
    yourdata = [(60001:70000)' submittal];
    dlmwrite(['./Submittals/' fname], yourdata, '-append',...
        'delimiter', ',');
end