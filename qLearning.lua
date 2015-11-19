--[[
Optimal by 2000 steps
.25 reward per step with epsilon of .1

--]]
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
alpha = .1
epsilon = .1
Q = torch.rand(10,2)

max_steps = 2e3
s = torch.random(9)
a = torch.random(2)
total_reward = 0
total_delta = 0
for t=1,max_steps do
    sPrime = P[s][a]
    if sPrime == goal then
        r = 1
        not_term = 0
    else
        r = 0
        not_term = 1
    end
    delta = r+not_term*gamma*torch.max(Q[sPrime]) - Q[s][a]
    Q[s][a] = Q[s][a] + alpha*delta
    if torch.rand(1)[1] > epsilon then
        _,a = torch.max(Q[sPrime],1)
        a = a[1]
    else
        a = torch.random(2)
    end
    if not_term == 0 then
        s = torch.random(9)
    else
        s = sPrime
    end
    total_reward = total_reward + r
    total_delta = total_delta + delta
    interval = 1e3
    if t%interval == 0 then
        print(total_reward/interval,total_delta,Q)
        total_delta = 0
        total_reward = 0
    end
end
    
