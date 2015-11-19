require 'gnuplot'
data = torch.load('datasets/gen_data2.t7')
gnuplot.hist(data.t_train)
