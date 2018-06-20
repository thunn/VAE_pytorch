from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

#Simple VAE architecture using linear encoder/decoder
class VAE_Linear(nn.Module):
    def __init__(self, hidden_dim=500, latent_dim=20):
        super(VAE_Linear, self).__init__()
        
        #encoder
        self.fc_e = nn.Linear(28*28, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        #decoder
        self.fc_d1 = nn.Linear(latent_dim, hidden_dim)
        self.fc_d2 = nn.Linear(hidden_dim, 28 * 28 )
        
    def encoder(self, x):
        ''' 
            encoder: q(z|x)
            input: x, output: mean, logvar
        '''
        x = F.relu(self.fc_e(x.view(-1, 28*28)))
        z_mean = self.fc_mean(x)
        z_logvar = self.fc_logvar(x)
        return z_mean, z_logvar
    
    def latent(self, z_mu, z_logvar):
        ''' 
            encoder: z = mu + sd * e
            input: mean, logvar. output: z
        '''
        sd = torch.exp(z_logvar * 0.5)
        e = Variable(torch.randn(sd.size()))
        z = e.mul(sd).add_(z_mu)
        return z 
    
    def decoder(self, z):
        '''
            decoder: p(x|z)
            input: z. output: x
        '''
        x = F.relu(self.fc_d1(z))
        x = F.sigmoid(self.fc_d2(x))
        return x.view(-1,1,28,28)

    def forward(self, x):
        z_mean, z_logvar = self.encoder(x)
        z = self.latent(z_mean, z_logvar)
        x_out = self.decoder(z)
        return x_out, z_mean, z_logvar
