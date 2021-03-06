-- Joost van Amersfoort - <joost@joo.st>
require 'sys'
require 'torch'
require 'nn'
require 'xlua'
require 'optim'
require 'nngraph'

--Packages necessary for SGVB
require 'Reparametrize'

--Custom Linear module to support different reset function
require 'LinearVA'

--For loading data files
require 'load'

require 'image'
require 'distributions'
data = load_mnist()

dim_input = data.train:size(2) 
hidden_units_encoder = 400
hidden_units_decoder = 400
decoder,dim_hidden = load_generator()
network = load_classifier()

--sample_size = 20
sample_size = 224
--generate 1 big grid of samples---------------------------------------
represent_sample = distributions.norm.qtl(torch.linspace(.01,.99,sample_size),0,1)
grid_sample = torch.zeros(sample_size^2,2)
for i = 1,sample_size do
    for j=1,sample_size do
        grid_sample[(i-1)*sample_size+j] = torch.Tensor{
                                    represent_sample[i],
                                    represent_sample[j]}
    end
end
--]]

--uniform sampling
--grid_sample = torch.randn(sample_size^2,dim_hidden)

--TODO:finish this option
--[[generate lots of minibatch size samples-----------------------------
sample_size = 6
num_batches = math.ceil(50000/(sample_size^2))
for b=1,num_batches do
    grid_sample = torch.zeros(sample_size^2,2)
    rand_offset = torch.rand(2):mul(1/sample_size)
    represent_sample = distributions.norm.qtl(torch.linspace(0,1-(1/sample_size),sample_size),0,1)
    for i = 1,sample_size do
        for j=1,sample_size do
            grid_sample[(i-1)*sample_size+j] = torch.Tensor{
                                        represent_sample[i]+rand_offset[1],
                                        represent_sample[j]+rand_offset[2]}
        end
    end
end
--]]



img = decoder:forward(grid_sample)

--classify samples-----------------------------------------
--class_probs = network:forward(data.x_train:double())
class_probs = network:forward(img)
_, classes = class_probs:max(2)
classes = classes:squeeze()
--
data = {}
data.x_train = img[{{1,50000},{}}]
data.t_train = classes[{{1,50000}}]
torch.save('datasets/gen_data.t7',data)
--]]
--[[display images------------------------------------------
print('got here')
side = math.sqrt(img[1]:size()[1])
for r = 1,sample_size do
    local row_image
    for c = 1,sample_size do
        --my_image = image.scale(img[torch.random(img:size()[1])]:reshape(side,side),800,600)
        my_image = img[(r-1)*sample_size+c]:reshape(side,side)
        if row_image then
            row_image = row_image:cat(my_image,2)
        else
            row_image = my_image:clone()
        end
    end
    if total_image then
        total_image = total_image:cat(row_image,1)
    else
        total_image = row_image:clone()
    end
end
image.display(total_image)
square_classes = classes:reshape(sample_size,sample_size):double()
img_side = total_image:size()[1]
scaled_classes = image.scale(square_classes,img_side,img_side)
image.display(scaled_classes)
--]]
