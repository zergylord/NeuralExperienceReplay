--[[
Optimal by 2000 steps
.25 reward per step with epsilon of .1

--]]
require 'nngraph'
require 'math'
require 'optim'
util = require 'util/my_torch_utils'
P = torch.zeros(10,2,10)
--left 1
P[1][1] = util.one_hot(10,10)
for i=2,10 do
    P[i][1] = util.one_hot(10,i-1)
end
--right 2
for i=1,9 do
    P[i][2] = util.one_hot(10,i+1)
end
P[10][2] = util.one_hot(10,1)
goal = 10
gamma = .9
epsilon = .1
--network setup
num_hid = 100
input = nn.Identity()()
hid = nn.ReLU()(nn.Linear(10,num_hid)(input))
out = nn.Linear(num_hid,2)(hid)

network = nn.gModule({input},{out})
parameters, gradients = network:getParameters()

config = {
    learningRate = 1e-4,
}

max_steps = 1e7
s = torch.random(9)
state = util.one_hot(10,s) 
a = torch.random(2)
total_reward = 0
total_loss = 0
local mse_crit = nn.MSECriterion()
for t=1,max_steps do
    state_prime = P[s][a]
    sPrime = util.get_ind(state_prime) 
    if sPrime == goal then
        r = 1
        not_term = 0
    else
        r = 0
        not_term = 1
    end
    qPrime = network:forward(state_prime):clone()
    q = network:forward(state):clone()
    target = q:clone()
    target[a] = r+not_term*gamma*qPrime:max()
    local opfunc = function(x)
        if x ~= parameters then
            parameters:copy(x)
        end

        network:zeroGradParameters()

        local loss = mse_crit:forward(q,target)
        local grad = mse_crit:backward(q,target)
        network:backward(state,grad)

        return loss, gradients
    end
    x, batchloss = optim.adagrad(opfunc, parameters, config)

    --select next action
    if torch.rand(1)[1] > epsilon then
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
    state = util.one_hot(10,s)

    total_reward = total_reward + r
    total_loss = total_loss + batchloss[1]
    interval = 1e5
    if t%interval == 0 then
        print(total_reward/interval,total_loss,gradients:norm(),q[2])
        total_reward = 0
        total_loss = 0
    end
end
    
