require 'gnuplot'
--data = torch.load('results.t7')
data = torch.load('results/agg.t7')
bucket_size = 1e4
num_buckets = data:size(3)/bucket_size
smoothed = torch.zeros(4,num_buckets)
ind_smoothed = torch.zeros(4,data:size(2),num_buckets)
num_episodes = data:size(2)
for m=1,4 do 
for b=1,num_buckets do
    local accum = 0
    for e=1,num_episodes do
        accum = accum + data[{{m},{e},{(b-1)*bucket_size+1,b*bucket_size}}]:sum()/(bucket_size*num_episodes)
        ind_smoothed[m][e][b]= data[{{m},{e},{(b-1)*bucket_size+1,b*bucket_size}}]:sum()/bucket_size
    end
    smoothed[m][b] = accum
end
end
gnuplot.plot({'dp',smoothed[1]},{'replay',smoothed[2]},{'represent',smoothed[3]},{'uniform',smoothed[4]})

