-- Joost networkn Amersfoort - <joost@joo.st>
require 'sys'
require 'torch'
require 'nn'
require 'xlua'
require 'optim'
require 'nngraph'
--Packages necessary for SGVB
require 'Reparametrize'
require 'KLDCriterion'

--Custom Linear module to support different reset function
require 'LinearVA'

--For loading data files
require 'load'
require 'image'
--data = load32()
--data = loadcatch()
data = load_mnist()
dim_input = data.train:size(2) 
dim_hid = 400
batchSize = 100

input = nn.Identity()()
hid = nn.ReLU()(nn.Linear(dim_input,dim_hid)(input))
output = nn.LogSoftMax()(nn.Linear(dim_hid,10)(hid))
network = nn.gModule({input},{output})

parameters, gradients = network:getParameters()

config = {
    learningRate = 0.03,
}

state = {}

nll_crit = nn.ClassNLLCriterion()

train_mode = true
if train_mode then
epoch = 0
while true do
    epoch = epoch + 1
    local loss = 0
    local time = sys.clock()
    local shuffle = torch.randperm(data.train:size(1))

    --Make sure batches are always batchSize
    local N = data.train:size(1) - (data.train:size(1) % batchSize)

    for i = 1, N, batchSize do
        xlua.progress(i+batchSize-1, data.train:size(1))

        local batch = torch.Tensor(batchSize,data.x_train:size(2))
        local target = torch.Tensor(batchSize)

        local k = 1
        for j = i,i+batchSize-1 do
            batch[k] = data.x_train[shuffle[j]]:clone() 
            target[k] = data.t_train[shuffle[j]]
            k = k + 1
        end

        local opfunc = function(x)
            if x ~= parameters then
                parameters:copy(x)
            end

            network:zeroGradParameters()

            f = network:forward(batch)
            local loss = nll_crit:forward(f, torch.add(target,1))
            local df_dw = nll_crit:backward(f, torch.add(target,1))
            network:backward(batch,df_dw)

            return loss, gradients
        end

        x, batchloss = optim.adagrad(opfunc, parameters, config, state)
        loss = loss + batchloss[1]
    end

    print("\nEpoch: " .. epoch .. " loss: " .. loss/N .. " time: " .. sys.clock() - time)

    if epoch % 2 == 0 then
        torch.save('save/classify_parameters.t7', parameters)
        torch.save('save/classify_state.t7', state)
    end
end

else
parameters:copy(torch.load('save/classify_parameters.t7'))
model_images = torch.load('save/image.t7')
batch = model_images[{{1,100},{}}]:double()
--batch = data.x_train[{{1,100},{}}]:double()
--target = data.t_train[{{1,100}}]
f = network:forward(batch)
ind = torch.random(100)
img = batch[ind]
side = math.sqrt(img:size()[1])
image.display(image.scale(img:reshape(side,side),800,600))
_,numeral = f[ind]:max(1)
print(numeral-1)
end
