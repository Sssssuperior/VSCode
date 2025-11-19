import torch.nn as nn
from .swin_transformer import swin_transformer
from .Transformer import Transformer
from .Transformer import token_Transformer
from .Decoder import Decoder,decoder_module,cross_attn
import torch
import torch.nn.functional as F


class ImageDepthNet(nn.Module):
    def __init__(self, args):
        super(ImageDepthNet, self).__init__()

        # VST Encoder
        self.rgb_backbone = swin_transformer(pretrained=True, args=args)
        
        self.mlp32 = nn.Sequential(
                nn.Linear(768, 384),
                nn.GELU(),
                nn.Linear(384, 384),)
                
        self.mlp16 = nn.Sequential(
                nn.Linear(384, 64),
                nn.GELU(),
                nn.Linear(64, 64),)
                
        self.norm1 = nn.LayerNorm(64)
        self.mlp1 = nn.Sequential(
            nn.Linear(64, 384),
            nn.GELU(),
            nn.Linear(384, 384),
        )
        self.fuse_32_16 = decoder_module(dim=384, token_dim=64, img_size=args.img_size, ratio=16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True)

        self.domain_promptR_1 = nn.Parameter(torch.randn(1, 1, 96))
        self.domain_promptR_2 = nn.Parameter(torch.randn(1, 1, 192))
        self.domain_promptR_3 = nn.Parameter(torch.randn(1, 1, 384))
        self.domain_promptR_4 = nn.Parameter(torch.randn(1, 1, 768))

        self.domain_promptD_1 = nn.Parameter(torch.randn(1, 1, 96))
        self.domain_promptD_2 = nn.Parameter(torch.randn(1, 1, 192))
        self.domain_promptD_3 = nn.Parameter(torch.randn(1, 1, 384))
        self.domain_promptD_4 = nn.Parameter(torch.randn(1, 1, 768))

        self.domain_promptT_1 = nn.Parameter(torch.randn(1, 1, 96))
        self.domain_promptT_2 = nn.Parameter(torch.randn(1, 1, 192))
        self.domain_promptT_3 = nn.Parameter(torch.randn(1, 1, 384))
        self.domain_promptT_4 = nn.Parameter(torch.randn(1, 1, 768))

        self.domain_promptV_1 = nn.Parameter(torch.randn(1, 1, 96))
        self.domain_promptV_2 = nn.Parameter(torch.randn(1, 1, 192))
        self.domain_promptV_3 = nn.Parameter(torch.randn(1, 1, 384))
        self.domain_promptV_4 = nn.Parameter(torch.randn(1, 1, 768))

        self.task_promptR_1 = nn.Parameter(torch.randn(1, 1, 96))
        self.task_promptR_2 = nn.Parameter(torch.randn(1, 1, 192))
        self.task_promptR_3 = nn.Parameter(torch.randn(1, 5, 384))
        self.task_promptR_4 = nn.Parameter(torch.randn(1, 10, 768))

        self.task_promptCR_1 = nn.Parameter(torch.randn(1, 1, 96))
        self.task_promptCR_2 = nn.Parameter(torch.randn(1, 1, 192))
        self.task_promptCR_3 = nn.Parameter(torch.randn(1, 5, 384))
        self.task_promptCR_4 = nn.Parameter(torch.randn(1, 10, 768))

        # VST Convertor
        self.transformer = Transformer(embed_dim=384, depth=4, num_heads=6, mlp_ratio=3.)

        # VST Decoder
        self.token_trans = token_Transformer(embed_dim=384, depth=4, num_heads=6, mlp_ratio=3.)
        self.decoder = Decoder(embed_dim=384, token_dim=64, depth=2, img_size=args.img_size)
        
        self.bs = args.batch_size

        # gate
        self.gate_domain = nn.ModuleList([
                 nn.Linear(embed_dim, 5)
                                        for embed_dim in [96,192,384,768]])
        self.gate_task = nn.ModuleList([
                 nn.Linear(embed_dim, 5)
                                        for embed_dim in [96,192,384,768]])
        
        # expert
        self.domain_expert1 = nn.ModuleList([nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        ) for embed_dim in [96,192,384,768]])

        self.domain_expert2 = nn.ModuleList([nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        ) for embed_dim in [96,192,384,768]])

        self.domain_expert3 = nn.ModuleList([nn.Sequential(
             nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        ) for embed_dim in [96,192,384,768]])

        self.domain_expert4 = nn.ModuleList([nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        ) for embed_dim in [96,192,384,768]])

        self.domain_expert5 = nn.ModuleList([nn.Sequential(
             nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        ) for embed_dim in [96,192,384,768]])
        
        self.task_expert1 = nn.ModuleList([nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        ) for embed_dim in [96,192,384,768]])

        self.task_expert2 = nn.ModuleList([nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        ) for embed_dim in [96,192,384,768]])

        self.task_expert3 = nn.ModuleList([nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        ) for embed_dim in [96,192,384,768]])

        self.task_expert4 = nn.ModuleList([nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        ) for embed_dim in [96,192,384,768]])

        self.task_expert5 = nn.ModuleList([nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        ) for embed_dim in [96,192,384,768]])
        
        
    def forward(self, image_Input, depth_Input, task):
        B, _, _, _ = image_Input.shape
        # VST Encoder
        # Training
        domain_prompt_pool1 = [self.domain_promptR_1, self.domain_promptD_1, self.domain_promptT_1, self.domain_promptV_1]
        domain_prompt_pool2 = [self.domain_promptR_2, self.domain_promptD_2, self.domain_promptT_2, self.domain_promptV_2]
        domain_prompt_pool3 = [self.domain_promptR_3, self.domain_promptD_3, self.domain_promptT_3, self.domain_promptV_3]
        domain_prompt_pool4 = [self.domain_promptR_4, self.domain_promptD_4, self.domain_promptT_4, self.domain_promptV_4]
        domain_prompt = [domain_prompt_pool1, domain_prompt_pool2, domain_prompt_pool3, domain_prompt_pool4]

        task_prompt_pool1 = [self.task_promptR_1, self.task_promptCR_1]
        task_prompt_pool2 = [self.task_promptR_2, self.task_promptCR_2]
        task_prompt_pool3 = [self.task_promptR_3, self.task_promptCR_3]
        task_prompt_pool4 = [self.task_promptR_4, self.task_promptCR_4]
        task_prompt = [task_prompt_pool1, task_prompt_pool2, task_prompt_pool3, task_prompt_pool4]

        expert_domain = [self.domain_expert1, self.domain_expert2, self.domain_expert3, self.domain_expert4, self.domain_expert5]
        expert_task = [self.task_expert1, self.task_expert2, self.task_expert3, self.task_expert4, self.task_expert5]

        num = [1, 1, 5, 10]
        
        
        if task == "RGB" or task == "CODRGB":
            output_rgb, output_depth = self.rgb_backbone(image_Input, depth_Input, domain_prompt, task_prompt, num, 
                                                   self.gate_domain, self.gate_task, expert_domain, expert_task)
            rgb_fea_1_4, rgb_fea_1_8, rgb_fea_1_16, rgb_fea_1_32 = output_rgb

            rgb_fea_1_32 = self.mlp32(rgb_fea_1_32)
            rgb_fea_1_16 = self.mlp16(rgb_fea_1_16)
            rgb_fea_1_16 = self.fuse_32_16(rgb_fea_1_32, rgb_fea_1_16)
            rgb_fea_1_16 = self.mlp1(self.norm1(rgb_fea_1_16))

        else:
            output_rgb, output_depth = self.rgb_backbone(image_Input, depth_Input, domain_prompt, task_prompt, num, 
                                                   self.gate_domain, self.gate_task, expert_domain, expert_task)

            rgb_fea_1_4, rgb_fea_1_8, rgb_fea_1_16, rgb_fea_1_32 = output_rgb
            _, _, depth_fea_1_16, depth_fea_1_32 = output_depth

            depth_fea_1_32 = self.mlp32(depth_fea_1_32)
            depth_fea_1_16 = self.mlp16(depth_fea_1_16)
            depth_fea_1_16 = self.fuse_32_16(depth_fea_1_32, depth_fea_1_16)
            depth_fea_1_16 = self.mlp1(self.norm1(depth_fea_1_16))

            rgb_fea_1_32 = self.mlp32(rgb_fea_1_32)
            rgb_fea_1_16 = self.mlp16(rgb_fea_1_16)
            rgb_fea_1_16 = self.fuse_32_16(rgb_fea_1_32, rgb_fea_1_16)
            rgb_fea_1_16 = self.mlp1(self.norm1(rgb_fea_1_16))

            rgb_fea_1_16 = (rgb_fea_1_16 + depth_fea_1_16)/2
           
        # VST Convertor
        rgb_fea_1_16 = self.transformer(rgb_fea_1_16)

        # VST Decoder
        fea_1_16, saliency_tokens, contour_tokens, fea_16, saliency_tokens_tmp, contour_tokens_tmp, saliency_fea_1_16, contour_fea_1_16, task_prompt, num1, task_prompt_total1 = self.token_trans(rgb_fea_1_16, task)
        outputs_saliency, outputs_contour, outputs_saliency_s, outputs_contour_s, fea_fg, fea_bg = self.decoder(fea_1_16, saliency_tokens, contour_tokens, fea_16, saliency_tokens_tmp, contour_tokens_tmp, saliency_fea_1_16, contour_fea_1_16, rgb_fea_1_8, rgb_fea_1_4, task_prompt, num1)

        return outputs_saliency, outputs_contour, outputs_saliency_s, outputs_contour_s