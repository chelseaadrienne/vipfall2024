# this class is responsible for training the model
class Unet(nn.Module):
    def __init__(self, *args, **kwargs):
        self.to_time_hiddens = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_cond_dim),
            nn.SiLU()
        )

        self.to_time_cond = nn.Sequential(
            nn.Linear(time_cond_dim, time_cond_dim)
        )

        self.to_time_tokens = nn.Sequential(
            nn.Linear(time_cond_dim, cond_dim * NUM_TIME_TOKENS),
            Rearrange('b (r d) -> b r d', r = NUM_TIME_TOKENS)
        )

        # gets our input images to the expected num of channels for the network
        self.init_conv = CrossEmbedLayer(channels, dim_out = dim, kernel_sizes = (3, 7, 15), stride = 1)

        # passing images into init_block; results in output tensor of the same size as the input
        self.init_block = ResnetBlock(current_dim, current_dim, cond_dim = layer_cond_dim, time_cond_dim = time_cond_dim, groups = groups)

        # passing images through resNet blocks
        self.resnet_blocks = nn.ModuleList(
            [
                ResnetBlock(current_dim, current_dim, time_cond,dim = time_cond_dim, groups = groups)
                for _ in range(layer_num_resnet_blocks)
            ]
        )

        # applies multi-headed attenton & passes output through a sequence of convolutions
        self.transformer_block = TransformerBlock(dim = current_dim, heads = ATTN_HEADS, dim_head = ATTN_DIM_HEAD)

        # downsampling images
        self.post_downsample = Downsample(current_dim, dim_out)

        # passing the images through more resNet blocks
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, cond_dim = con_dim, time_cond_dim = time_cond_dim, groups = resnet_groups[-1])
        
        self.mid_attn = EinopsToAndFrom('b c h w', 'b (h w) c', Residual(Attention(mid_dim, heads = ATTN_HEADS, dim_head = ATTN_DIM_HEAD))) if attend_at_middle else None
        
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, cond_dim = cond_dim, time_cond_dim = time_cond_dim, groups = resnet_groups[-1])

    # forward method - 
    def forward(self, *args, **kwargs):
        time_hiddens = self.to_time_hiddens(time)

        t = self.to_time_cond(time_hiddens)
        time_tokens = self.to_time_tokens(time_hiddens)

        x = self.init_conv(x)

        x = init_block(x, t, c)

        hiddens = []

        for resnet_block in self.resnet_blocks:
            x = resnet_block(x, t)
            hiddens.append(x)

        x = self.transformer_block(x)
        hiddens.append(x)

        # downsampling images
        x = post_downsample(x)

        # passing the images through more resNet blocks
        x = self.mid_block1(x, t, c)
        if exists(self.mid_attn):
            x = self.mid_attn(x)
        x = self.mid_block2(x, t, c)

    # method is a convolution
    def Downsample(dim, dim_out = None):
        dim_out = default(dim_out, dim)
        return nn.Conv2d(dim, dim_out, kernel_size = 4, stride = 2, padding = 1)
    
    # mirror-inverse of the downsampling trajectory; 
    def Upsample(dim, dim_out = None):
        dim_out = default(dim_out, dim)

        return nn.Sequential(
            nn.Upsample(scale_factor = 2, mode = 'nearest'), nn.Conv2d(dim, dim_out, 3, padding = 1)
        )    
    