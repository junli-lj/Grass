function out = myLeakyReLU(x)
if x > 0
    out = x;
else
    out = 0.2*x;
end

end

