function CreateSubmittal(input, nn, fname)
    a = nn.test(input);
    a = OneHotDecode(setMax(a)) - 1;
    WriteToFile(fname, a);
end