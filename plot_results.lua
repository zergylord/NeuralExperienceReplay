require 'gnuplot'
data = torch.load('results.t7')
bucket_size = 1e4
num_buckets = data:size(3)/bucket_size
--smoothed = torch.zeros(4,num_buckets)
smoothed = torch.zeros(4,data:size(2),num_buckets)
for m=1,4 do 
for b=1,num_buckets do
    local accum = 0
    for e=1,data:size(2) do
        --accum = accum + data[{{m},{e},{b,b+bucket_size-1}}]:sum()
        smoothed[m][e][b]= data[{{m},{e},{b,b+bucket_size-1}}]:sum()/bucket_size
    end
    --smoothed[m][b] = accum
end
end
