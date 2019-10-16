function name = saveName(n, l)%(1,2)
name = num2str(n);
for i=1:l-length(name)
    name = ['0' name];%[01]
end;
end
