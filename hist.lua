require 'gnuplot'
data = torch.load('datasets/gen_data.t7')
gnuplot.hist(data.t_train)
