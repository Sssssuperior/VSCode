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

        self.linearR1 = nn.Linear(96, 384)
        self.linearR2 = nn.Linear(192,384)
        self.linearR4 = nn.Linear(768,384)

        self.linearD1 = nn.Linear(96, 384)
        self.linearD2 = nn.Linear(192, 384)
        self.linearD4 = nn.Linear(768, 384)

        self.linearT1 = nn.Linear(96, 384)
        self.linearT2 = nn.Linear(192, 384)
        self.linearT4 = nn.Linear(768, 384)

        self.linearV1 = nn.Linear(96, 384)
        self.linearV2 = nn.Linear(192, 384)
        self.linearV4 = nn.Linear(768, 384)

        self.linearS1 = nn.Linear(96, 384)
        self.linearS2 = nn.Linear(192, 384)
        self.linearS4 = nn.Linear(768, 384)

        self.linearC1 = nn.Linear(96, 384)
        self.linearC2 = nn.Linear(192, 384)
        self.linearC4 = nn.Linear(768, 384)
        
        self.linearR_low = nn.Linear(384 * 4, 384)
        self.linearD_low = nn.Linear(384 * 4, 384)
        self.linearT_low = nn.Linear(384 * 4, 384)
        self.linearV_low = nn.Linear(384 * 4, 384)
        self.linearS_low = nn.Linear(384 * 4, 384)
        self.linearC_low = nn.Linear(384 * 4, 384)

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

        self.linear1 = nn.Linear(96, 384)
        self.linear2 = nn.Linear(192, 384)
        self.linear3 = nn.Linear(384, 384)
        self.linear4 = nn.Linear(768, 384)
        self.down = nn.Linear(384 * 4, 384)
        self.all = nn.Linear(384 * 5, 384)
        self.all_e = nn.Linear(384 * 3, 384)
        
        
    def forward(self, image_Input, depth_Input):

        B, _, _, _ = image_Input.shape
        # VST Encoder
        # Training
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
        
        output_rgb, output_depth, rgb_prompt_d, rgb_prompt_t, domain_prompt_d, domain_prompt_t = self.rgb_backbone(image_Input, depth_Input[:-4, :, :, :], domain_prompt, task_prompt, num, 
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

        rgb_fea = rgb_fea_1_16[-4:, :, :]  # 2, HW, C
        rgb_fea_1_16 = (rgb_fea_1_16[:-4, :, :] + depth_fea_1_16)/2
        rgb_fea_1_16 = torch.cat((rgb_fea_1_16, rgb_fea), dim=0)
            #rgb_fea_1_16 = torch.cat((rgb_fea_1_16, rgb_fea), dim=0)
        # VST Convertor
        rgb_fea_1_16 = self.transformer(rgb_fea_1_16)
        # rgb_fea_1_16 [B, 14*14, 384]

        # VST Decoder
        fea_1_16, saliency_tokens, contour_tokens, fea_16, saliency_tokens_tmp, contour_tokens_tmp, saliency_fea_1_16, contour_fea_1_16, task_prompt, num1, task_prompt_total1, task_prompt1 = self.token_trans(rgb_fea_1_16)
        # fea_1_16 [B, 1 + 14*14 + 1, 384]
        # saliency_tokens [B, 1, 384]
        # contour_fea_1_16 [B, 14*14, 384]
        # contour_tokens [B, 1, 384]
        outputs_saliency, outputs_contour, outputs_saliency_s, outputs_contour_s, fea_fg, fea_bg = self.decoder(fea_1_16, saliency_tokens, contour_tokens, fea_16, saliency_tokens_tmp, contour_tokens_tmp, saliency_fea_1_16, contour_fea_1_16, rgb_fea_1_8, rgb_fea_1_4, task_prompt, num1)

        domain_promptR = torch.mean(self.linearR_low(torch.cat([prompt for prompt in
                                    [self.linearR1(self.domain_promptR_1), self.linearR2(self.domain_promptR_2), self.domain_promptR_3, self.linearR4(self.domain_promptR_4)]], dim=2)), dim=1, keepdim=True)
        domain_promptD = torch.mean(self.linearD_low(torch.cat([prompt for prompt in
                                    [self.linearD1(self.domain_promptD_1), self.linearD2(self.domain_promptD_2), self.domain_promptD_3, self.linearD4(self.domain_promptD_4)]], dim=2)), dim=1, keepdim=True)
        domain_promptT = torch.mean(self.linearT_low(torch.cat([prompt for prompt in
                                    [self.linearT1(self.domain_promptT_1), self.linearT2(self.domain_promptT_2), self.domain_promptT_3, self.linearT4(self.domain_promptT_4)]], dim=2)), dim=1, keepdim=True)
        domain_promptV = torch.mean(self.linearV_low(torch.cat([prompt for prompt in
                                    [self.linearV1(self.domain_promptV_1), self.linearV2(self.domain_promptV_2), self.domain_promptV_3, self.linearV4(self.domain_promptV_4)]], dim=2)), dim=1, keepdim=True)
        domain_prompt_total = [domain_promptR, domain_promptD, domain_promptT, domain_promptV]
        
        task_promptS = self.linearS_low(torch.cat([prompt for prompt in
                                    [self.linearS1(self.task_promptR_1), self.linearS2(self.task_promptR_2), torch.mean(self.task_promptR_3, dim=1, keepdim=True), torch.mean(self.linearS4(self.task_promptR_4), dim=1, keepdim=True)]], dim=2))
        task_promptC = self.linearC_low(torch.cat([prompt for prompt in
                                    [self.linearC1(self.task_promptCR_1), self.linearC2(self.task_promptCR_2), torch.mean(self.task_promptCR_3, dim=1, keepdim=True), torch.mean(self.linearC4(self.task_promptCR_4), dim=1, keepdim=True)]], dim=2))
        task_prompt_total2 = [task_promptS, task_promptC]

        # aggregate prompt
        rgb_d = self.down(torch.cat([prompt for prompt in [self.linear1(rgb_prompt_d[0]), self.linear2(rgb_prompt_d[1]), self.linear3(rgb_prompt_d[2]), self.linear4(rgb_prompt_d[3])]], dim=-1))
        rgb_t = self.down(torch.cat([prompt for prompt in [self.linear1(rgb_prompt_t[0]), self.linear2(rgb_prompt_t[1]), self.linear3(torch.mean(rgb_prompt_t[2], dim=1, keepdim=True)), self.linear4(torch.mean(rgb_prompt_t[3], dim=1, keepdim=True))]], dim=-1))
        domain_d = self.down(torch.cat([prompt for prompt in [self.linear1(domain_prompt_d[0]), self.linear2(domain_prompt_d[1]), self.linear3(domain_prompt_d[2]), self.linear4(domain_prompt_d[3])]], dim=-1))
        domain_t = self.down(torch.cat([prompt for prompt in [self.linear1(domain_prompt_t[0]), self.linear2(domain_prompt_t[1]), self.linear3(torch.mean(domain_prompt_t[2], dim=1, keepdim=True)), self.linear4(torch.mean(domain_prompt_t[3], dim=1, keepdim=True))]], dim=-1))
        
        prompt_aggregated1 = self.all(torch.cat((rgb_d.squeeze(1)[:-4], rgb_t[:-4], domain_d.squeeze(1), domain_t, task_prompt1.mean(dim=1,keepdim=True)[:-4]), dim=-1))
        prompt_aggregated2 = self.all_e(torch.cat((rgb_d.squeeze(1)[-4:], rgb_t[-4:], task_prompt1.mean(dim=1,keepdim=True)[-4:]), dim=-1))
        prompt_aggregated = torch.cat((prompt_aggregated1, prompt_aggregated2), dim=0)
        return outputs_saliency, outputs_contour, outputs_saliency_s, outputs_contour_s, domain_prompt_total, task_prompt_total2, task_prompt_total1, fea_fg, fea_bg, prompt_aggregated