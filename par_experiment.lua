modes = {'dp','replay','represent','uniform'}
max_steps = 5e4
m = torch.random(#modes)
print(m)
training_mode = modes[m]
dofile('train_mnist_maze.lua')
torch.save(modes[m] .. '_results' .. torch.rand(1)[1] .. '.t7',reward_log)
