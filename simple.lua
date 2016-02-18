require 'gnuplot'
require 'optim'
require 'nngraph'
--Packages necessary for SGVB
require 'Reparametrize'
require 'KLDCriterion'

--Custom Linear module to support different reset function
require 'LinearVA'
in_dim = 1
hid_dim = 2 --double the dimension of latent space
mb_dim = 320

--encoder
e_input = nn.Identity()()
e_hid = nn.Tanh()(nn.LinearVA(in_dim,hid_dim)(e_input))
e_mu = nn.LinearVA(hid_dim,hid_dim/2)(e_hid)
e_sigma = nn.LinearVA(hid_dim,hid_dim/2)(e_hid)
encoder = nn.gModule({e_input},{e_mu,e_sigma})

--decoder
d_in = nn.Identity()()
d_hid = nn.Tanh()(nn.LinearVA(hid_dim/2,hid_dim)(d_in))
d_out = nn.LinearVA(hid_dim,in_dim)(d_hid)
decoder = nn.gModule({d_in},{d_out})

--full network
input = nn.Identity()()
repar = nn.Reparametrize(hid_dim/2)(encoder(input))
output = decoder(repar)
network = nn.gModule({input},{output})

--network:forward(torch.rand(in_dim))

--Binary cross entropy term
BCE = nn.BCECriterion()
BCE.sizeAverage = false
MSE = nn.MSECriterion()
KLD = nn.KLDCriterion()

w, dw = network:getParameters()

config = {
    learningRate = -1e-4,
}

opfunc = function(x)
    if x ~= w then
        w:copy(x)
    end

    network:zeroGradParameters()

    mb = torch.randn(mb_dim,in_dim)
    --mask = torch.rand(mb_dim,in_dim):gt(.5)
    --mb[mask] = .2


    out = network:forward(mb)
    local err = - MSE:forward(out, mb)
    local do_dw = MSE:backward(out, mb):mul(-1)

    network:backward(mb,do_dw)

    local KLDerr = KLD:forward(encoder.output, mb)
    local de_dw = KLD:backward(encoder.output, mb)

    local scale = 2e-3
    encoder:backward(mb,{de_dw[1]:mul(scale),de_dw[2]:mul(scale)})

    local lowerbound = err  + KLDerr

    return lowerbound, dw

end
refresh = 1e3
plot1 = gnuplot.figure()
plot2 = gnuplot.figure()
plot3 = gnuplot.figure()
for t = 1,1e5 do
    local lowerbound = 0
    local time = sys.clock()
    x, mblowerbound = optim.adam(opfunc, w, config)
    lowerbound = lowerbound + mblowerbound[1]
    if t % refresh == 0 then
        print("step: " .. t .. " Lowerbound: " .. lowerbound .. " time: " .. sys.clock() - time)
        train_mb = mb:clone()
        train_out = out:clone()
        train_hid = {encoder.output[1]:clone()[{{},1}],encoder.output[2]:clone()[{{},1}],'+'}
        train_toterr,train_KLDerr = KLD:forward(encoder.output, train_mb)
        train_KLDerr = train_KLDerr:mean(2)

        test_mb = torch.rand(mb_dim,in_dim):add(-.5):mul(20)
        test_out = network:forward(test_mb)
        test_hid = {encoder.output[1]:clone()[{{},1}],encoder.output[2]:clone()[{{},1}],'+'}
        test_toterr,test_KLDerr = KLD:forward(encoder.output, test_mb)
        test_KLDerr = test_KLDerr:mean(2)

        print(train_toterr,test_toterr)
        --Data Fit
        gnuplot.figure(plot1)
        gnuplot.plot({train_mb[{{},1}],train_out[{{},1}],'+'},{test_mb[{{},1}],test_out[{{},1}],'+'},{torch.linspace(-10,10,100),torch.linspace(-10,10,100)})
        gnuplot.axis{-10,10,-10,10}
        --Mu vs Sigma (desired=0,0)
        gnuplot.figure(plot2)
        gnuplot.plot(train_hid,test_hid)
        --data vs KL error
        gnuplot.figure(plot3)
        gnuplot.plot({train_mb[{{},1}],train_KLDerr[{{},1}],'+'},{test_mb[{{},1}],test_KLDerr[{{},1}],'+'})
        lowerbound = 0
    end
end
