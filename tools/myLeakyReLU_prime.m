function out = myLeakyReLU_prime(x)
if x > 0
    out = ones(size(x));
else
    out = 0.2*ones(size(x));
end

end

