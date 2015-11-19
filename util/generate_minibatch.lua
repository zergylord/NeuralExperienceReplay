function generate_minibatch()
    sample_size = math.ceil(math.sqrt(batch_size))
    local img
    if train_mode == 'represent'  then
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
