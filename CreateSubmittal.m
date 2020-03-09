function CreateSubmittal(input, nn, fname)
    a = nn.test(input);
    a = a(:, 1:10000) + a(:, 10001:20000) + a(:,20001:30000);
    a = OneHotDecode(setMax(a)) - 1;
    WriteToFile(fname, a);
end