function result = OneHotEncoder(y)
    n_classes = max(y) + 1;

    result = zeros( size( y, 1 ), n_classes );

    % assuming class labels start from one
    for i = 1:n_classes
        rows = y + 1 == i;
        result(rows, i) = 1;
    end
end