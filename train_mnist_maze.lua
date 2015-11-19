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
require 'distributions'
util = require 'util/my_torch_utils'
require 'util/generate_minibatch'
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
num_hid = 100 
input = nn.Identity()()
hid = nn.ReLU()(nn.Linear(state_dim,num_hid)(input))
out = nn.Linear(num_hid,2)(hid)

network = nn.gModule({input},{out})
parameters, gradients = network:getParameters()

--Replay setup-----
batch_size = 36
--
training_mode = 'replay'
max_steps = 1e7
--]]
burn_in = 100
if training_mode == 'dp' then
    batch_size = 20
    replay_size = batch_size
    state_hist = torch.zeros(replay_size,state_dim)
    s_hist= torch.zeros(replay_size)
    sPrime_hist= torch.zeros(replay_size)
    action_hist = torch.zeros(replay_size,1)
    state_prime_hist = torch.zeros(replay_size,state_dim)
    not_term_hist = torch.ones(replay_size,1)
    reward_hist = torch.zeros(replay_size,1)
    for i = 1,10 do
        s_hist[i] = i
        s_hist[10+i] = i
        action_hist[i][1] = 1
        action_hist[10+i][1] = 2
        sPrime_hist[i] = P[i][1]
        if sPrime_hist[i] == goal then
            not_term_hist[i][1] = 0
            reward_hist[i][1] = 1
        end
        sPrime_hist[10+i] = P[i][2]
        if sPrime_hist[10+i] == goal then
            not_term_hist[10+i][1] = 0
            reward_hist[10+i][1] = 1
        end
    end
elseif training_mode == 'replay' then
    replay_size = 1000
    state_hist = torch.zeros(replay_size,state_dim)
    action_hist = torch.zeros(replay_size,1)
    state_prime_hist = torch.zeros(replay_size,state_dim)
    s_hist = torch.zeros(replay_size,1)
    sPrime_hist = torch.zeros(replay_size,1)
    not_term_hist = torch.ones(replay_size,1)
    reward_hist = torch.zeros(replay_size,1)
    replay_ind = 1
elseif training_mode == 'uniform' or training_mode == 'represent'  then
    replay_size = batch_size
    state_hist = torch.zeros(replay_size,state_dim)
    s_hist= torch.zeros(replay_size)
    sPrime_hist= torch.zeros(replay_size)
    action_hist = torch.zeros(replay_size,1)
    state_prime_hist = torch.zeros(replay_size,state_dim)
    not_term_hist = torch.ones(replay_size,1)
    reward_hist = torch.zeros(replay_size,1)
    generator = load_generator()
    classifier = load_classifier()
end
-----------------
require 'load'
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

s = torch.random(9)
state = get_mnist_digit(s) 
a = torch.random(2)
total_reward = 0
total_loss = 0
local mse_crit = nn.MSECriterion()
interval = 1e4
state_counts = torch.zeros(10)
reward_log = torch.zeros(max_steps)
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
    reward_log[t] = r
    if training_mode == 'replay' then
        --add to replay----------
        state_hist[replay_ind]:copy(state)
        s_hist[replay_ind][1] = s
        action_hist[replay_ind][1] = a
        state_prime_hist[replay_ind]:copy(state_prime)
        sPrime_hist[replay_ind][1] = sPrime
        reward_hist[replay_ind][1] = r
        not_term_hist[replay_ind][1] = not_term
    elseif training_mode == 'dp' then
        for i=1,10 do
            state_hist[i]:copy(get_mnist_digit(i))
            state_prime_hist[i]:copy(get_mnist_digit(sPrime_hist[i]))
            state_hist[i+10]:copy(get_mnist_digit(i))
            state_prime_hist[i+10]:copy(get_mnist_digit(sPrime_hist[i+10]))
        end
    elseif training_mode == 'uniform' or training_mode == 'represent' then
        state_hist,s_hist = generate_minibatch()
        for i=1,batch_size do
            action_hist[i] = torch.random(2)
            sPrime_hist[i] = P[s_hist[i][1]][action_hist[i][1]]
            if sPrime_hist[i] == goal then
                not_term_hist[i] = 0
                reward_hist[i] = 1
            else
                not_term_hist[i] = 1
                reward_hist[i] = 0
            end
            state_prime_hist[i]:copy(get_mnist_digit(sPrime_hist[i]))
        end
    end
    --update network----------
    local opfunc = function(x)
        if x ~= parameters then
            parameters:copy(x)
        end

        network:zeroGradParameters()

        local loss
        if training_mode == 'replay' then
            bind = torch.zeros(replay_size,1)
            local chosen = torch.randperm(math.min(t,replay_size))[{{1,batch_size}}]
            for i = 1,batch_size do 
                bind[chosen[i]][1] = 1
            end
            bind = bind:byte()

            state_bind = bind:expandAs(state_hist)
            qPrime_hist = network:forward(state_prime_hist[state_bind]:reshape(batch_size,state_dim)):clone()
            q = network:forward(state_hist[state_bind]:reshape(batch_size,state_dim)):clone()
            target = q:clone()
            local expected_return = reward_hist[bind] + qPrime_hist:max(2):mul(gamma):cmul(not_term_hist[bind])
            target:scatter(2,action_hist[bind]:reshape(batch_size,1):long(),
                    expected_return:reshape(batch_size,1))

            loss = mse_crit:forward(q,target)
            local grad = mse_crit:backward(q,target)
            network:backward(state_hist[state_bind]:reshape(batch_size,state_dim),grad)
        else
            qPrime_hist = network:forward(state_prime_hist):clone()
            q = network:forward(state_hist):clone()
            target = q:clone()
            local expected_return = reward_hist + qPrime_hist:max(2):mul(gamma):cmul(not_term_hist)
            target:scatter(2,action_hist:long(),expected_return)
            loss = mse_crit:forward(q,target)
            local grad = mse_crit:backward(q,target)
            network:backward(state_hist,grad)
        end

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
    if training_mode == 'replay'  then
        replay_ind = (replay_ind % (replay_size))+1
    end
    if t%interval == 0 then
        print(t,total_reward/interval,sys.clock()-time,total_loss,gradients:norm(),state_counts)
        time = sys.clock()
        state_counts:zero()
        total_reward = 0
        total_loss = 0
    end
end
    
