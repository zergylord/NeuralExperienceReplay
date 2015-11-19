--[[
Optimal by 2000 steps
.25 reward per step with epsilon of .1

--]]
time = sys.clock()
require 'nngraph'
require 'math'
require 'optim'
require 'image'
require 'load'
util = require 'util/my_torch_utils'
require 'distributions'
P = torch.zeros(10,2)
--left 1
P[1][1] = 10
for i=2,10 do
    P[i][1] = i-1
end
--right 2
for i=1,9 do
    P[i][2] = i+1
end
P[10][2] = 1

goal = 10
gamma = .9
epsilon = .1
--network setup
state_dim = 28*28
local num_hid = 100 
local input = nn.Identity()()
local hid = nn.ReLU()(nn.Linear(state_dim,num_hid)(input))
local out = nn.Linear(num_hid,2)(hid)

network = nn.gModule({input},{out})
parameters, gradients = network:getParameters()

--Replay setup-----
batch_size = 36
--round to closest square
sample_size = math.ceil(math.sqrt(batch_size))
batch_size = math.pow(sample_size,2)
--uniform sampling
representative = true
state_gen = torch.zeros(batch_size,state_dim)
action_gen = torch.zeros(batch_size,1)
state_prime_gen = torch.zeros(batch_size,state_dim)
not_term_gen = torch.ones(batch_size,1)
reward_gen = torch.zeros(batch_size,1)

generator = load_generator()
classifier = load_classifier()

--TODO:return classes for forward model to use
function generate_minibatch()
    local img
    if representative then
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
        local uniform_sample = torch.randn(sample_size^2,2)
        img = generator:forward(uniform_sample)
    end
    local class_probs = classifier:forward(img)
    local _, classes = class_probs:max(2)
    return img,classes
end
-----------------------------
--data = load_mnist()
data = torch.load('datasets/gen_data.t7')
data.t_train = data.t_train:add(-1)
data_by_number = {}
for n=1,10 do
    mask = data.t_train:eq(n-1):reshape(50000,1)
    data_by_number[n] = data.x_train[mask:expandAs(data.x_train)]:reshape(mask:sum(),28*28):clone()
end
function get_mnist_digit(numeral)
    ind = torch.random(data_by_number[numeral]:size()[1])
    return data_by_number[numeral][ind]:double()
end

config = {
    learningRate = 1e-4,
}

max_steps = 1e7
s = torch.random(9)
state = get_mnist_digit(s) 
a = torch.random(2)
total_reward = 0
total_loss = 0
local mse_crit = nn.MSECriterion()
interval = 1e4
state_counts = torch.zeros(10)
burn_in = 32
for t=1,max_steps do
    sPrime = P[s][a]
    state_prime = get_mnist_digit(sPrime) 
    state_counts[sPrime] = state_counts[sPrime] + 1
    if sPrime == goal then
        r = 1
        not_term = 0
    else
        r = 0
        not_term = 1
    end
    --update network----------
    local opfunc = function(x)
        if x ~= parameters then
            parameters:copy(x)
        end

        network:zeroGradParameters()

        state_gen,s_gen = generate_minibatch()
        for i =1,batch_size do
            action_gen[i] = torch.random(2)
            local next_state = P[s_gen[i][1]][action_gen[i][1]]
            if next_state == goal then
                not_term_gen[i] = 0
                reward_gen[i] = 1
            else
                not_term_gen[i] = 1
                reward_gen[i] = 0
            end
            state_prime_gen[i] = get_mnist_digit(next_state)
        end
        local qPrime = network:forward(state_prime_gen):clone()
        local q = network:forward(state_gen):clone()
        local target = q:clone()
        local expected_return = reward_gen + qPrime:max(2):mul(gamma):cmul(not_term_gen)

        target:scatter(2,action_gen:long(),
                expected_return)

        local loss = mse_crit:forward(q,target)
        local grad = mse_crit:backward(q,target)
        network:backward(state_gen,grad)

        return loss, gradients
    end
    if t > burn_in then
        x, batchloss = optim.adagrad(opfunc, parameters, config)
        total_loss = total_loss + batchloss[1]
    end
    ---------------------------

    --select next action
    if torch.rand(1)[1] > epsilon then
        qPrime = network:forward(state_prime)
        _,a = torch.max(qPrime,1)
        a = a[1]
    else
        a = torch.random(2)
    end
    if not_term == 0 then
        s = torch.random(9)
    else
        s = sPrime
    end
    state = get_mnist_digit(s)

    total_reward = total_reward + r
    if t%interval == 0 then
        print(t,total_reward/interval,sys.clock()-time,total_loss,gradients:norm(),state_counts)
        time = sys.clock()
        state_counts:zero()
        total_reward = 0
        total_loss = 0
    end
end
    
