time = sys.clock()
a = {'asdf','bar'}
b = 1
for i=1,1e6 do
    ind = torch.random(2)
    --
    if a[ind] == 'asdf' then
        b= b+1
    end
    --]]
    b=b+1
end
print(sys.clock()-time)
