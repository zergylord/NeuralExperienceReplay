local dim
local function aux(accum,ind_accum)
    if #ind_accum == dim then
        for i=1,dim do
            accum[i] = represent_sample[ind_accum[i]] +rand_offset[i]
        end
    else
        for i=1,sample_size do
            table.insert(ind_accum,i)
            accum[i] = aux(torch.zeros(accum[i]:size()),ind_accum)
            table.remove(ind_accum)
        end
    end
    return accum
end
function generate_minibatch(d)
    dim = d
    if dim > 2 then
        sample_size = 2
    else
        sample_size = math.ceil(math.sqrt(batch_size))
    end
    local img
    if training_mode == 'represent'  then
        rand_offset = torch.rand(dim):mul(1/sample_size)
        represent_sample = distributions.norm.qtl(torch.linspace(0,1-(1/sample_size),sample_size),0,1)
        local dim_sizes = {}
        for i=1,dim do
            dim_sizes[i] = sample_size
        end
        dim_sizes[dim+1] = dim
        local grid_sample = aux(torch.zeros(unpack(dim_sizes)),{})

        img = generator:forward(grid_sample:reshape(sample_size^dim,dim))
    else --uniform sampling
        local uniform_sample = torch.randn(batch_size,dim)
        img = generator:forward(uniform_sample)
    end
    local class_probs = classifier:forward(img)
    local _, classes = class_probs:max(2)
    return img,classes
end
--[[
function generate_minibatch()
    sample_size = math.ceil(math.sqrt(batch_size))
    local img
    if training_mode == 'represent'  then
        local rand_offset = torch.rand(2):mul(1/sample_size)
        local represent_sample = distributions.norm.qtl(torch.linspace(0,1-(1/sample_size),sample_size),0,1)
        local grid_sample = torch.zeros(sample_size^2,2)
        for i = 1,sample_size do
            for j=1,sample_size do
                grid_sample[(i-1)*sample_size+j] = torch.Tensor{
                                            represent_sample[i]+rand_offset[1],
                                            represent_sample[j]+rand_offset[2]}
            end
        end
        
        img = generator:forward(grid_sample)
    else --uniform sampling
        local uniform_sample = torch.randn(batch_size,2)
        img = generator:forward(uniform_sample)
    end
    local class_probs = classifier:forward(img)
    local _, classes = class_probs:max(2)
    return img,classes
end
--]]
