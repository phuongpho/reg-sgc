from dgl.nn.pytorch.conv import SGConv

class regSGConv(SGConv):
    def __init__(self,
                in_feats,
                out_feats,
                L1 = 0.0,
                L2 = 0.0,
                L3 = 0.0,
                ortho = False,
                k = 1,
                cached=False,
                bias=True,
                norm=None,
                allow_zero_in_degree=False):
        super().__init__(
                in_feats,
                out_feats,
                k=k,
                cached=cached,
                bias=bias,
                norm=norm,
                allow_zero_in_degree=allow_zero_in_degree)
        
        # Check regularization mode
        reg_zero = [L1 == 0.0, L2 == 0.0, L3 == 0.0]
        if all(reg_zero):
                print("Initializing SGC without regularization !!!")
        
        self.L1 = float(L1)
        self.L2 = float(L2)
        self.L3 = float(L3)
        self.ortho = ortho
        
        message = f"Initializing regularized SGC: L1 = {self.L1}, L2 = {self.L2}, L3 = {self.L3}"
        if ortho:
                print(message + '\nUsing orthogonality constraint for L3')
        else:
                print(message + '\nUsing opposing constraint for L3')