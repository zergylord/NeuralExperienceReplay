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
epsilon = 0
--network setup
num_hid = 100
input = nn.Identity()()
hid = nn.ReLU()(nn.Linear(10,num_hid)(input))
out = nn.Linear(num_hid,2)(hid)

network = nn.gModule({input},{out})
parameters, gradients = network:getParameters()

--Replay setup
state_hist = torch.zeros(20,10)
action_hist = torch.zeros(20,1)
state_prime_hist = torch.zeros(20,10)
not_term_hist = torch.ones(20,1)
reward_hist = torch.zeros(20,1)
for i = 1,10 do
    local temp_state = util.one_hot(10,i)
    state_hist[i] = temp_state:clone()
    state_hist[10+i] = temp_state:clone()
    action_hist[i][1] = 1
    action_hist[10+i][1] = 2
    state_prime_hist[i] = P[i][1]:clone()
    if util.get_ind(state_prime_hist[i]) == goal then
        not_term_hist[i][1] = 0
        reward_hist[i][1] = 1
    end
    state_prime_hist[10+i] = P[i][2]:clone()
    if util.get_ind(state_prime_hist[10+i]) == goal then
        not_term_hist[10+i][1] = 0
        reward_hist[10+i][1] = 1
    end
end

-------------

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
interval = 1e4
state_counts = torch.zeros(10)
for t=1,max_steps do
    state_prime = P[s][a]
    sPrime = util.get_ind(state_prime) 
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
        
        qPrime_hist = network:forward(state_prime_hist):clone()
        q = network:forward(state_hist):clone()
        target = q:clone()
        local expected_return = reward_hist + qPrime_hist:max(2):mul(gamma):cmul(not_term_hist)
        if t%interval == 0 then
            print(q[{{1,10},{}}])
            print(qPrime_hist[{{1,10},{}}])
            _,policy = q[{{1,10},{}}]:max(2)
            print(policy)
            --print(expected_return)
        end
        target:scatter(2,action_hist:long(),expected_return)

        local loss = mse_crit:forward(q,target)
        local grad = mse_crit:backward(q,target)
        network:backward(state,grad)

        return loss, gradients
    end
    x, batchloss = optim.adagrad(opfunc, parameters, config)
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
    state = util.one_hot(10,s)

    total_reward = total_reward + r
    total_loss = total_loss + batchloss[1]
    if t%interval == 0 then
        print(total_reward/interval,total_loss,gradients:norm(),state_counts)
        state_counts:zero()
        total_reward = 0
        total_loss = 0
    end
end
    
