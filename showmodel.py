from torchsummary import summary
from linknetmodel import linknet

model = linknet(19)

model.cuda()

summary(model, (3, 360, 480))
