function result = OneHotDecode(value)
    n = size(value, 1);
    result = zeros(size(value, 2), 1);
    for i = 1:n
       row = value(i, :);
       result(row == 1) = i;
    end
end