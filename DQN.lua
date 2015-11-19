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

--Replay setup
replay_size = 1000
state_hist = torch.zeros(replay_size,10)
action_hist = torch.zeros(replay_size,1)
state_prime_hist = torch.zeros(replay_size,10)
not_term_hist = torch.ones(replay_size,1)
reward_hist = torch.zeros(replay_size,1)
replay_ind = 1
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
burn_in = 32
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
    --add to replay----------
    state_hist[replay_ind]:copy(state)
    action_hist[replay_ind][1] = a
    state_prime_hist[replay_ind]:copy(state_prime)
    reward_hist[replay_ind][1] = r
    not_term_hist[replay_ind][1] = not_term
    --update network----------
    local opfunc = function(x)
        if x ~= parameters then
            parameters:copy(x)
        end

        network:zeroGradParameters()

        bind = torch.zeros(replay_size,1)
        local chosen = torch.randperm(math.min(t,replay_size))[{{1,32}}]
        for i = 1,32 do 
            bind[chosen[i]][1] = 1
        end
        bind = bind:byte()
        state_bind = bind:expandAs(state_hist)
        qPrime_hist = network:forward(state_prime_hist[state_bind]:reshape(32,10)):clone()
        q = network:forward(state_hist[state_bind]:reshape(32,10)):clone()
        target = q:clone()
        local expected_return = reward_hist[bind] + qPrime_hist:max(2):mul(gamma):cmul(not_term_hist[bind])
        if t%interval == 0 then
            print(q[{{1,10},{}}])
            print(qPrime_hist[{{1,10},{}}])
            _,policy = q[{{1,10},{}}]:max(2)
            print(policy)
            --print(expected_return)
        end
        --print(action_hist[bind])
        target:scatter(2,action_hist[bind]:reshape(32,1):long(),expected_return:reshape(32,1))

        local loss = mse_crit:forward(q,target)
        local grad = mse_crit:backward(q,target)
        network:backward(state,grad)

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
    state = util.one_hot(10,s)

    total_reward = total_reward + r
    replay_ind = (replay_ind % (replay_size))+1
    if t%interval == 0 then
        print(total_reward/interval,total_loss,gradients:norm(),state_counts)
        state_counts:zero()
        total_reward = 0
        total_loss = 0
    end
end
    
