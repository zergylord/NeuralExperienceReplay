--Packages necessary for SGVB
require 'Reparametrize'

--Custom Linear module to support different reset function
require 'LinearVA'

function load32()
    data = {}
    data.train = torch.load('datasets/train_32x32.t7', 'ascii').data
    data.test = torch.load('datasets/test_32x32.t7', 'ascii').data

    --Convert training data to floats
    data.train = data.train:double()
    data.test = data.test:double()

    --Rescale to 0..1 and invert
    data.train:div(255):resize(60000,1024)
    data.test:div(255):resize(10000,1024)

    return data
end

function loadfreyfaces(path)
    require 'hdf5'
    local f = hdf5.open(path, 'r')
    local data = {}
    data.train = f:read('train'):all():double()
    data.test = f:read('test'):all():double()

    return data
end

function loadcatch()
    data = {}
    data.train = torch.load('datasets/foo.t7')
    data.test = torch.load('datasets/bar.t7')
    return data
end

function load_mnist()
    require 'hdf5'
    local f = hdf5.open('datasets/mnist.hdf5')
    local data = f:read():all()
    --[[
    local data.train = f:read('x_train'):all():double()
    local data.train_labels = f:read('t_train'):all():double()
    local data.test = f:read('x_valid'):all():double()
    local data.test_labels = f:read('t_valid'):all():double()
    --]]
    data.train = data.x_train
    data.test = data.x_test
    return data
end

function load_generator()
    --The VAE model-------------------------------

    local dim_input = 28*28
    local dim_hidden = 2
    local hidden_units_encoder = 400
    local hidden_units_decoder = 400
    --Encoding layer
    local encoder = nn.Sequential()
    encoder:add(nn.LinearVA(dim_input,hidden_units_encoder))
    encoder:add(nn.Tanh())

    local z = nn.ConcatTable()
    z:add(nn.LinearVA(hidden_units_encoder, dim_hidden))
    z:add(nn.LinearVA(hidden_units_encoder, dim_hidden))

    encoder:add(z)

    local va = nn.Sequential()
    va:add(encoder)

    --Reparametrization step
    va:add(nn.Reparametrize(dim_hidden))

    --Decoding layer
    local decoder = nn.Sequential()
    decoder:add(nn.LinearVA(dim_hidden, hidden_units_decoder))
    decoder:add(nn.Tanh())
    decoder:add(nn.LinearVA(hidden_units_decoder, dim_input))
    decoder:add(nn.Sigmoid())
    va:add(decoder)

    local gen_parameters = va:getParameters()

    gen_parameters:copy(torch.load('save/2_parameters.t7'))
    return decoder
end
function load_classifier()
    --the classifier model -------------------------------
    local dim_input = 28*28
    local dim_hid = 400
    local input = nn.Identity()()
    local hid = nn.ReLU()(nn.Linear(dim_input,dim_hid)(input))
    local output = nn.LogSoftMax()(nn.Linear(dim_hid,10)(hid))
    local network = nn.gModule({input},{output})

    local class_parameters = network:getParameters()
    class_parameters:copy(torch.load('save/2_classify_parameters.t7'))
    return network
end 
