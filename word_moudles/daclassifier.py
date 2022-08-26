import torch
import torch.nn as nn
import torch.nn.functional as F

class ReverseGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()
def reverse_grad(x):
    func = ReverseGrad.apply
    return func(x)

def half_reverse_grad(x):
    x_shape = x.shape
    lastdim = len(x_shape)-1
    func = ReverseGrad.apply
    return torch.cat([func(torch.narrow(x,lastdim,0,x_shape[-1]//2)),
                      torch.narrow(x,lastdim,x_shape[-1]//2,x_shape[-1]-x_shape[-1]//2)], dim=lastdim)

class ScaleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output*ctx.scale, None
def scale_grad(x,sacle):
    return ScaleGrad.apply(x,sacle)

### MLP
class MLP_full(nn.Module):
    def __init__(self, ndf = None,classes=1):
        super(MLP_full, self).__init__()

        self.ndf = ndf
        self.main = nn.Sequential(
            # Z goes into a linear of size: ndf
            nn.Linear(isize, self.ndf//2),
            nn.PReLU(),
            nn.Linear(self.ndf//2, self.ndf//4),
            nn.PReLU(),
            nn.Linear(self.ndf//4, self.ndf//8),
            nn.PReLU(),
            nn.Linear(self.ndf//8, classes)
        )

    def forward(self, input):
        input = input.transpose(0, 1).contiguous().view(input.shape[1],-1)
        return self.main(input)

class MLP_sum(MLP_full):
    def forward(self, input):
        input = input.sum(dim=0)
        return self.main(input)

class MLP_word(nn.Module):
    def __init__(self, isize, ndf = None, classes=1):
        super(MLP_word, self).__init__()

        if ndf == None:
            ndf = isize
        self.ndf = ndf
        self.main = None
        self.main = nn.Sequential(
            # Z goes into a linear of size: ndf
            nn.Linear(isize, self.ndf//4),
            nn.PReLU(),
            nn.Linear(self.ndf//4, classes)
        )
        self.isize = isize

    def forward(self, input):
        input_shape = input.shape
        return self.main(input)

class MLP_att(nn.Module):
    def __init__(self, embed_dim, attention_hidden = None, ndf = None, classes=1):
        super(MLP_att, self).__init__()
        self.embed_dim = embed_dim
        if attention_hidden == None:
            attention_hidden = embed_dim
        self.attention_hidden = attention_hidden
        self.attlinear1 = nn.Linear(self.embed_dim, self.attention_hidden, bias=False)
        self.attlinear2 = nn.Linear(self.attention_hidden, 1, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=0)
        if ndf == None:
            ndf = embed_dim
        self.ndf = ndf

        self.main = nn.Sequential(
            # Z goes into a linear of size: ndf
            nn.Linear(self.embed_dim, self.ndf//2),
            nn.PReLU(),
            nn.Linear(self.ndf//2, self.ndf//4),
            nn.PReLU(),
            nn.Linear(self.ndf//4, self.ndf//8),
            nn.PReLU(),
            nn.Linear(self.ndf//8, classes)
        )

    def forward(self, input):
        input_len, batch_size, _ = input.size()
        ## Attention
        att = input.view(input_len * batch_size, self.embed_dim)
        att = self.tanh(self.attlinear1(att))
        att = self.attlinear2(att).view(input_len, batch_size)
        att = self.softmax(att)
        input = torch.einsum('lb,lbd->bd',(att,input))
        ## MLP
        return self.main(input)

class MLP_LossMask(nn.Module):
    def __init__(self, embed_dim, attention_hidden = None, classes=1):
        super(MLP_LossMask, self).__init__()
        self.embed_dim = embed_dim
        if attention_hidden == None:
            attention_hidden = embed_dim
        self.attention_hidden = attention_hidden
        self.attlinear1 = nn.Linear(self.embed_dim, self.attention_hidden, bias=False)
        self.attlinear2 = nn.Linear(self.attention_hidden, 1, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=0)
        self.main = nn.Sequential(
            nn.PReLU(),
            nn.Linear(self.embed_dim, classes)
        )

    def forward(self, input):
        input = input.detach()
        input_len, batch_size, _ = input.size()
        ## Attention
        att = input.view(input_len * batch_size, self.embed_dim)
        att = self.tanh(self.attlinear1(att))
        att = self.attlinear2(att).view(input_len, batch_size)
        att = self.softmax(att)
        input = torch.einsum('lb,lbd->bd',(att,input))
        ## MLP
        return self.main(input), att

class MLP_simple(nn.Module):
    def __init__(self, embed_dim, attention_hidden = None, n_nodes = 4, ndf = None, classes=1):
        super(MLP_simple, self).__init__()
        self.embed_dim = embed_dim
        if attention_hidden == None:
            attention_hidden = embed_dim
        self.attention_hidden = attention_hidden
        self.attlinear1 = nn.Linear(self.embed_dim, self.attention_hidden, bias=False)
        self.attlinear2 = nn.Linear(self.attention_hidden, 1, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=0)
        if ndf == None:
            ndf = embed_dim
        self.ndf = ndf
        self.main = nn.Sequential(
            # Z goes into a linear of size: ndf
            nn.Dropout(p=0.5),
            nn.Linear(self.embed_dim, classes)
        )

    def forward(self, input):
        input_len, batch_size, embed_dim = input.size()
        input = input[:,:,:embed_dim//2]+input[:,:,embed_dim//2:]
        ## Attention
        att = input.view(input_len * batch_size, self.embed_dim)
        att = self.tanh(self.attlinear1(att))
        att = self.attlinear2(att).view(input_len, batch_size)
        att = self.softmax(att)
        input = torch.einsum('lb,lbd->bd',(att,input))
        ## MLP
        return self.main(input)


### LSTM
class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim=None, num_layers=2, bidirectional=True, classes=1):

        super(LSTMClassifier, self).__init__()

        self.embedding_dim = embedding_dim
        if hidden_dim == None:
            hidden_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, bidirectional=bidirectional)
        self.dropout_layer = nn.Dropout(p=0.2)
        self.bidirectional = bidirectional
        if bidirectional:
            self.hidden2out = nn.Sequential(
                        nn.Linear(hidden_dim*2, classes),
                        )
        else:
            self.hidden2out = nn.Sequential(
                        nn.Linear(hidden_dim, classes),
                        )

    def init_hidden(self, batch_size, device):
        if self.bidirectional:
            return (torch.randn(self.num_layers*2, batch_size, self.hidden_dim).to(device),
                    torch.randn(self.num_layers*2, batch_size, self.hidden_dim).to(device))
        else:
            return (torch.randn(self.num_layers, batch_size, self.hidden_dim).to(device),
                    torch.randn(self.num_layers, batch_size, self.hidden_dim).to(device))

    def forward(self, input):
        batchsize = input.shape[1]
        hidden = self.init_hidden(input.shape[1], input.device)
        outputs, (ht, ct) = self.lstm(input, hidden)

        # ht is the last hidden state of the sequences
        # ht = (num_layers * num_directions, batch_size, hidden_dim)
        output = ht.view(self.num_layers, -1, input.shape[1], self.hidden_dim)[-1,:,:,:]
        output = output.transpose(0, 1).contiguous().view(batchsize,-1)
        #output = self.dropout_layer(output)
        output = self.hidden2out(output)

        return output


class V2Net(nn.Module):
    def __init__(self):
        super(V2Net, self).__init__()
        self.net1 = None
        self.net2 = None

    def forward(self, input):
        input_shape = input.shape
        out1 = self.net1(input[:,:,:input_shape[2]//2])
        out2 = self.net2(input[:,:,input_shape[2]//2:])
        if len(out1.shape)==2:
            out1=out1.unsqueeze(0)
            out2=out2.unsqueeze(0)
        return torch.cat([out1,out2],dim=0)


class LSTMClassifierV2(V2Net):

    def __init__(self, embedding_dim, hidden_dim=None, num_layers=2, bidirectional=True, classes=1):
        super(LSTMClassifierV2, self).__init__()
        self.net1 = LSTMClassifier(embedding_dim//2,hidden_dim,num_layers,bidirectional, classes=classes)
        self.net2 = LSTMClassifier(embedding_dim-embedding_dim//2,hidden_dim,num_layers,bidirectional, classes=classes)

class MLP_att_V2(nn.Module):

    def __init__(self, embedding_dim, classes=1):
        super(MLP_att_V2, self).__init__()
        self.net1 = MLP_att(embedding_dim//2,classes=classes)
        self.net2 = MLP_att(embedding_dim-embedding_dim//2,classes=classes)

class MLP_word_V2(nn.Module):

    def __init__(self, embedding_dim, classes=1):
        super(MLP_word_V2, self).__init__()
        self.net1 = MLP_word(embedding_dim//2, classes=classes)
        self.net2 = MLP_word(embedding_dim-embedding_dim//2, classes=classes)

class GradWrapper(nn.Module):
    def __init__(self, args, classifer):
        super(GradWrapper, self).__init__()
        self.classifer = classifer
        self.args = args

    def forward(self, input):
        input = input['encoder_out']
        if self.args.damethod == "pure_adv":
            input = reverse_grad(input)
        elif self.args.damethod == "partial_adv":
            input_shape = input.shape
            advportion = 5
            # Use input_shape[1] leads to better performance for LSMT LOL DKW
            input_1 = reverse_grad(input[:,:,:(input_shape[2]*advportion)//10])
            input_2 = input[:,:,(input_shape[2]*advportion)//10:]
            # input_1 = reverse_grad(input[:,:,:25])
            # input_2 = input[:,:,-25:]
            # input_2 = scale_grad(input[:,:,(input_shape[2]*advportion)//10:],0.01)
            input = torch.cat([input_1,input_2],dim=2)
        return self.classifer(input)


### Logistics Regression
class LogReg(nn.Module):
    def __init__(self, ndf = None, classes=1):
        super(LogReg, self).__init__()
        self.ndf = ndf
        self.main = nn.Sequential(
            nn.Linear(ndf,classes)
        )

    def forward(self, input):
        return self.main(input)


def builddaclassifier(args):
    classes = args.domain_nums
    if classes == 2:
        classes = 1
    if args.daclassifier in ["MLP_att","MLP"]:
        classifer = MLP_att(args.encoder_embed_dim, classes=classes)
    elif args.daclassifier in ["MLP_att_V2"]:
        classifer = MLP_att_V2(args.encoder_embed_dim, classes=classes)
    elif args.daclassifier in ["MLP_word"]:
        classifer = MLP_word(args.encoder_embed_dim, classes=classes)
    elif args.daclassifier in ["MLP_word_V2"]:
        classifer = MLP_word_V2(args.encoder_embed_dim, classes=classes)
    elif args.daclassifier in ["MLP_simple"]:
        classifer = MLP_simple(args.encoder_embed_dim//2, classes=classes)
    elif args.daclassifier in ["MLP_full"]:
        classifer = MLP_full(args.encoder_embed_dim*128, classes=classes)
    elif args.daclassifier in ["MLP_sum"]:
        classifer = MLP_sum(args.encoder_embed_dim, classes=classes)
    elif args.daclassifier in ["LSTMClassifier", "LSTM"]:
        classifer = LSTMClassifier(args.encoder_embed_dim,100,1, classes=classes)
    elif args.daclassifier in ["LSTMClassifierV2", "LSTMV2"]:
        classifer = LSTMClassifierV2(args.encoder_embed_dim,100,1, classes=classes)
    return GradWrapper(args,classifer)

def build_bayesclassifier(args, dim):
    classes = args.domain_nums
    if classes == 2:
        classes = 1
    if args.bayesclassifier in ["Logistics"]:
        return LogReg(dim, classes=classes)
