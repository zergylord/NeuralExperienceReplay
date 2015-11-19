-- Joost van Amersfoort - <joost@joo.st>
require 'sys'
require 'torch'
require 'nn'
require 'xlua'
require 'optim'

--Packages necessary for SGVB
require 'Reparametrize'
require 'KLDCriterion'

--Custom Linear module to support different reset function
require 'LinearVA'

--For loading data files
require 'load'

require 'image'
require 'distributions'
data = load_mnist()

dim_input = data.train:size(2) 
dim_hidden = 2
hidden_units_encoder = 400
hidden_units_decoder = 400
batchSize = 100
--The model

--Encoding layer
encoder = nn.Sequential()
encoder:add(nn.LinearVA(dim_input,hidden_units_encoder))
encoder:add(nn.Tanh())

z = nn.ConcatTable()
z:add(nn.LinearVA(hidden_units_encoder, dim_hidden))
z:add(nn.LinearVA(hidden_units_encoder, dim_hidden))

encoder:add(z)

va = nn.Sequential()
va:add(encoder)

--Reparametrization step
va:add(nn.Reparametrize(dim_hidden))

--Decoding layer
--[[
va:add(nn.LinearVA(dim_hidden, hidden_units_decoder))
va:add(nn.Tanh())
va:add(nn.LinearVA(hidden_units_decoder, dim_input))
va:add(nn.Sigmoid())
--]]
decoder = nn.Sequential()
decoder:add(nn.LinearVA(dim_hidden, hidden_units_decoder))
decoder:add(nn.Tanh())
decoder:add(nn.LinearVA(hidden_units_decoder, dim_input))
decoder:add(nn.Sigmoid())
va:add(decoder)
--Binary cross entropy term
BCE = nn.BCECriterion()
BCE.sizeAverage = false
KLD = nn.KLDCriterion()

parameters, gradients = va:getParameters()

parameters:copy(torch.load('save/parameters.t7'))
batch = data.train[{{1,100},{}}]:double()
--output = va:forward(batch)
--img = image.scale(output[1]:reshape(28,28),800,600)
--image.display(img)

--img = decoder:forward(torch.randn(100,dim_hidden))
sample_size = 30
represent_sample = distributions.norm.qtl(torch.linspace(.01,.99,sample_size),0,1)
grid_sample = torch.zeros(sample_size^2,2)
for i = 1,sample_size do
    for j=1,sample_size do
        grid_sample[(i-1)*sample_size+j] = torch.Tensor{
                                    represent_sample[i],
                                    represent_sample[j]}
    end
end
--grid_sample = represent_sample:reshape(10,1):repeatTensor(1,dim_hidden)
img = decoder:forward(grid_sample)
--display images
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
