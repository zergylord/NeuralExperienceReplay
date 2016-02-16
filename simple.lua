require 'optim'
require 'nngraph'
--Packages necessary for SGVB
require 'Reparametrize'
require 'KLDCriterion'

--Custom Linear module to support different reset function
require 'LinearVA'
in_dim = 1
hid_dim = 40 --double the dimension of latent space
mb_dim = 1

--encoder
e_input = nn.Identity()()
e_hid = nn.Tanh()(nn.LinearVA(in_dim,hid_dim)(e_input))
e_mu = nn.LinearVA(hid_dim,hid_dim/2)(e_hid)
e_sigma = nn.LinearVA(hid_dim,hid_dim/2)(e_hid)
encoder = nn.gModule({e_input},{e_mu,e_sigma})

--decoder
d_in = nn.Identity()()
d_hid = nn.Tanh()(nn.LinearVA(hid_dim/2,hid_dim)(d_in))
d_out = nn.Sigmoid()(nn.LinearVA(hid_dim,in_dim)(d_hid))
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
KLD = nn.KLDCriterion()

w, dw = network:getParameters()

config = {
    learningRate = -1e-2,
}

opfunc = function(x)
    if x ~= w then
        w:copy(x)
    end

    network:zeroGradParameters()

    mb = torch.rand(mb_dim,in_dim):mul(.8)
    --mask = torch.rand(mb_dim,in_dim):gt(.5)
    --mb[mask] = .2


    out = network:forward(mb)
    local err = - BCE:forward(out, mb)
    local do_dw = BCE:backward(out, mb):mul(-1)

    network:backward(mb,do_dw)

    local KLDerr = KLD:forward(encoder.output, mb)
    local de_dw = KLD:backward(encoder.output, mb)

    encoder:backward(mb,de_dw)

    local lowerbound = err  + KLDerr

    return lowerbound, dw

end
refresh = 1e3
for t = 1,1e5 do
    local lowerbound = 0
    local time = sys.clock()
    x, mblowerbound = optim.adam(opfunc, w, config)
    lowerbound = lowerbound + mblowerbound[1]
    if t % refresh == 0 then
        print("step: " .. t .. " Lowerbound: " .. lowerbound .. " time: " .. sys.clock() - time)
        print(mb[1],out[1])
        lowerbound = 0
    end
end
