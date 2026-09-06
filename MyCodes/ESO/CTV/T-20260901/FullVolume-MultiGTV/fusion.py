import torch
from torch import nn
class MultiSliceImageFusion(nn.Module):
 def __init__(self):super().__init__();self.convs=nn.ModuleList();self.alpha=nn.ParameterList()
 def materialize(self,features,sizes):
  while len(self.convs)<len(features):
   c=int(features[len(self.convs)].shape[-1]);dev=features[len(self.convs)].device;self.convs.append(nn.Conv2d(3*c,c,1).to(dev));self.alpha.append(nn.Parameter(torch.zeros((),device=dev)))
 def forward(self,previous,current,following,sizes):
  self.materialize(current,sizes);out=[]
  for i,(a,b,c,(h,w)) in enumerate(zip(previous,current,following,sizes)):
   def grid(x):return x.permute(1,2,0).reshape(x.shape[1],x.shape[2],h,w)
   context=self.convs[i](torch.cat((grid(a),grid(b),grid(c)),1));x=grid(b)+self.alpha[i]*context;out.append(x.reshape(x.shape[0],x.shape[1],h*w).permute(2,0,1))
  return out
class MultiSliceGTVPromptFusion(nn.Module):
 def __init__(self):super().__init__();self.beta=nn.Parameter(torch.zeros(()));self.conv=None
 def materialize(self,dense):
  if self.conv is None:self.conv=nn.Conv2d(3*int(dense.shape[1]),int(dense.shape[1]),1).to(dense.device)
 def forward(self,previous,current,following):self.materialize(current);return current+self.beta*self.conv(torch.cat((previous,current,following),1))