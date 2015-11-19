require 'torch'
require 'image'
require 'math'
img = torch.load('image.t7')
side = math.sqrt(img[1]:size()[1])
for r = 1,10 do
    local row_image
    for c = 1,10 do
        --my_image = image.scale(img[torch.random(img:size()[1])]:reshape(side,side),800,600)
        my_image = img[torch.random(img:size()[1])]:reshape(side,side)
        if row_image then
            row_image = row_image:cat(my_image)
        else
            row_image = my_image:clone()
        end
    end
    if total_image then
        total_image = total_image:cat(row_image,1)
    else
        total_image = row_image:clone()
    end
end
image.display(total_image)
