require 'pprint'
modes = {'dp','replay','represent','uniform'}
max_steps = 5e4
trials = 3
results = torch.zeros(#modes,trials,max_steps)
for t = 1,trials do
    print('-----------------------------',t)
    for m = 1,#modes do
        print('---------------',modes[m])
        training_mode = modes[m]
        dofile('train_mnist_maze.lua')
        results[m][t] = reward_log:clone() 
    end
end
torch.save('results.t7',results)
