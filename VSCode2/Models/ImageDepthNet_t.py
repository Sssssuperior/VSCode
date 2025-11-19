import torch.nn as nn
from .swin_transformer_t import swin_transformer_t
from .Transformer import Transformer
from .Transformer import token_Transformer
from .Decoder import Decoder,decoder_module
import torch
import torch.nn.functional as F


class ImageDepthNet_t(nn.Module):
    def __init__(self, args):
        super(ImageDepthNet_t, self).__init__()

        # VST Encoder
        self.rgb_backbone = swin_transformer_t(pretrained=True, args=args)
        
        # fuse 1/32 and 1/16 features
        self.mlp32 = nn.Sequential(
                nn.Linear(args.encoder_dim[3], args.encoder_dim[2]),
                nn.GELU(),
                nn.Linear(args.encoder_dim[2], args.encoder_dim[2]),)
                
        self.mlp16 = nn.Sequential(
                nn.Linear(args.encoder_dim[2], args.dim),
                nn.GELU(),
                nn.Linear(args.dim, args.dim),)
                
        self.norm1 = nn.LayerNorm(args.dim)
        self.mlp1 = nn.Sequential(
            nn.Linear(args.dim, args.embed_dim),
            nn.GELU(),
            nn.Linear(args.embed_dim, args.embed_dim),
        )
        self.fuse_32_16 = decoder_module(dim=args.embed_dim, token_dim=args.dim, img_size=args.img_size, ratio=16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True)

        self.norm = nn.LayerNorm(args.embed_dim * 2)
        self.mlp_s = nn.Sequential(
            nn.Linear(args.embed_dim * 2, args.embed_dim),
            nn.GELU(),
            nn.Linear(args.embed_dim, args.embed_dim),
        )
        
        # domain prompt for RGB DEPTH THERMAL FLOW
        self.domain_promptR_1 = nn.Parameter(torch.randn(1, args.domain_num[0], args.encoder_dim[0]))
        self.domain_promptR_2 = nn.Parameter(torch.randn(1, args.domain_num[1], args.encoder_dim[1]))
        self.domain_promptR_3 = nn.Parameter(torch.randn(1, args.domain_num[2], args.encoder_dim[2]))
        self.domain_promptR_4 = nn.Parameter(torch.randn(1, args.domain_num[3], args.encoder_dim[3]))
        
        self.domain_promptD_1 = nn.Parameter(torch.randn(1, args.domain_num[0], args.encoder_dim[0]))
        self.domain_promptD_2 = nn.Parameter(torch.randn(1, args.domain_num[1], args.encoder_dim[1]))
        self.domain_promptD_3 = nn.Parameter(torch.randn(1, args.domain_num[2], args.encoder_dim[2]))
        self.domain_promptD_4 = nn.Parameter(torch.randn(1, args.domain_num[3], args.encoder_dim[3]))

        self.domain_promptT_1 = nn.Parameter(torch.randn(1, args.domain_num[0], args.encoder_dim[0]))
        self.domain_promptT_2 = nn.Parameter(torch.randn(1, args.domain_num[1], args.encoder_dim[1]))
        self.domain_promptT_3 = nn.Parameter(torch.randn(1, args.domain_num[2], args.encoder_dim[2]))
        self.domain_promptT_4 = nn.Parameter(torch.randn(1, args.domain_num[3], args.encoder_dim[3]))

        self.domain_promptV_1 = nn.Parameter(torch.randn(1, args.domain_num[0], args.encoder_dim[0]))
        self.domain_promptV_2 = nn.Parameter(torch.randn(1, args.domain_num[1], args.encoder_dim[1]))
        self.domain_promptV_3 = nn.Parameter(torch.randn(1, args.domain_num[2], args.encoder_dim[2]))
        self.domain_promptV_4 = nn.Parameter(torch.randn(1, args.domain_num[3], args.encoder_dim[3]))

        # task prompt for RGB CODRGB
        self.task_promptR_1 = nn.Parameter(torch.randn(1, args.task_num[0], args.encoder_dim[0]))
        self.task_promptR_2 = nn.Parameter(torch.randn(1, args.task_num[1], args.encoder_dim[1]))
        self.task_promptR_3 = nn.Parameter(torch.randn(1, args.task_num[2], args.encoder_dim[2]))
        self.task_promptR_4 = nn.Parameter(torch.randn(1, args.task_num[3], args.encoder_dim[3]))

        self.task_promptCR_1 = nn.Parameter(torch.randn(1, args.task_num[0], args.encoder_dim[0]))
        self.task_promptCR_2 = nn.Parameter(torch.randn(1, args.task_num[1], args.encoder_dim[1]))
        self.task_promptCR_3 = nn.Parameter(torch.randn(1, args.task_num[2], args.encoder_dim[2]))
        self.task_promptCR_4 = nn.Parameter(torch.randn(1, args.task_num[3], args.encoder_dim[3]))

        self.num_task = args.task_num
        self.num_domain = args.domain_num
        self.num_deco = args.task_deco_num

        # VST Convertor
        self.transformer = Transformer(embed_dim=384, depth=4, num_heads=6, mlp_ratio=3.)

        # VST Decoder
        self.token_trans = token_Transformer(embed_dim=384, depth=4, num_heads=6, mlp_ratio=3.)
        self.decoder = Decoder(embed_dim=384, token_dim=64, depth=2, img_size=args.img_size)
        
        self.bs = args.batch_size
        
        # prompt discrimination loss
        self.linearR1 = nn.Linear(args.encoder_dim[0], args.embed_dim)
        self.linearR2 = nn.Linear(args.encoder_dim[1], args.embed_dim)
        self.linearR4 = nn.Linear(args.encoder_dim[3], args.embed_dim)
        
        self.linearD1 = nn.Linear(args.encoder_dim[0], args.embed_dim)
        self.linearD2 = nn.Linear(args.encoder_dim[1], args.embed_dim)
        self.linearD4 = nn.Linear(args.encoder_dim[3], args.embed_dim)
        
        self.linearT1 = nn.Linear(args.encoder_dim[0], args.embed_dim)
        self.linearT2 = nn.Linear(args.encoder_dim[1], args.embed_dim)
        self.linearT4 = nn.Linear(args.encoder_dim[3], args.embed_dim)
        
        self.linearV1 = nn.Linear(args.encoder_dim[0], args.embed_dim)
        self.linearV2 = nn.Linear(args.encoder_dim[1], args.embed_dim)
        self.linearV4 = nn.Linear(args.encoder_dim[3], args.embed_dim)
        
        self.linearS1 = nn.Linear(args.encoder_dim[0], args.embed_dim)
        self.linearS2 = nn.Linear(args.encoder_dim[1], args.embed_dim)
        self.linearS4 = nn.Linear(args.encoder_dim[3], args.embed_dim)
        
        self.linearC1 = nn.Linear(args.encoder_dim[0], args.embed_dim)
        self.linearC2 = nn.Linear(args.encoder_dim[1], args.embed_dim)
        self.linearC4 = nn.Linear(args.encoder_dim[3], args.embed_dim)
        
        self.linearR_low = nn.Linear(args.embed_dim*4, args.embed_dim)
        self.linearD_low = nn.Linear(args.embed_dim*4, args.embed_dim)
        self.linearT_low = nn.Linear(args.embed_dim*4, args.embed_dim)
        self.linearV_low = nn.Linear(args.embed_dim*4, args.embed_dim)
        self.linearS_low = nn.Linear(args.embed_dim*4, args.embed_dim)
        self.linearC_low = nn.Linear(args.embed_dim*4, args.embed_dim)

    def forward(self, image_Input, depth_Input):

        B, _, _, _ = image_Input.shape
        # VST Encoder
        # D T V CV SR CR
        task_promptR1 = torch.cat((self.task_promptR_1.repeat(3 * self.bs // 6, 1, 1), self.task_promptCR_1.repeat(self.bs // 6, 1, 1), self.task_promptR_1.repeat(self.bs // 6, 1, 1), self.task_promptCR_1.repeat(self.bs // 6, 1, 1)), dim=0)
        task_promptR2 = torch.cat((self.task_promptR_2.repeat(3 * self.bs // 6, 1, 1), self.task_promptCR_2.repeat(self.bs // 6, 1, 1), self.task_promptR_2.repeat(self.bs // 6, 1, 1), self.task_promptCR_2.repeat(self.bs // 6, 1, 1)), dim=0)
        task_promptR3 = torch.cat((self.task_promptR_3.repeat(3 * self.bs // 6, 1, 1), self.task_promptCR_3.repeat(self.bs // 6, 1, 1), self.task_promptR_3.repeat(self.bs // 6, 1, 1), self.task_promptCR_3.repeat(self.bs // 6, 1, 1)), dim=0)
        task_promptR4 = torch.cat((self.task_promptR_4.repeat(3 * self.bs // 6, 1, 1), self.task_promptCR_4.repeat(self.bs // 6, 1, 1), self.task_promptR_4.repeat(self.bs // 6, 1, 1), self.task_promptCR_4.repeat(self.bs // 6, 1, 1)), dim=0)
        task_prompt = [task_promptR1, task_promptR2, task_promptR3, task_promptR4]
        
        domain_promptR1 = self.domain_promptR_1.repeat(self.bs, 1, 1)
        domain_promptR2 = self.domain_promptR_2.repeat(self.bs, 1, 1)
        domain_promptR3 = self.domain_promptR_3.repeat(self.bs, 1, 1)
        domain_promptR4 = self.domain_promptR_4.repeat(self.bs, 1, 1)
        domain_promptR = [domain_promptR1, domain_promptR2, domain_promptR3, domain_promptR4]
        
        # RGB prompt
        rgb_fea_1_4 , rgb_fea_1_8, rgb_fea_1_16, rgb_fea_1_32 = self.rgb_backbone(image_Input, task_prompt, self.num_task, domain_promptR)
        
        domain_prompt1 = torch.cat([prompt.repeat(self.bs // 6, 1, 1) for prompt in
                                    [self.domain_promptD_1, self.domain_promptT_1, self.domain_promptV_1, self.domain_promptV_1]], dim=0)
        domain_prompt2 = torch.cat([prompt.repeat(self.bs // 6, 1, 1) for prompt in
                                    [self.domain_promptD_2, self.domain_promptT_2, self.domain_promptV_2, self.domain_promptV_2]], dim=0)
        domain_prompt3 = torch.cat([prompt.repeat(self.bs // 6, 1, 1) for prompt in
                                    [self.domain_promptD_3, self.domain_promptT_3, self.domain_promptV_3, self.domain_promptV_3]], dim=0)
        domain_prompt4 = torch.cat([prompt.repeat(self.bs // 6, 1, 1) for prompt in
                                    [self.domain_promptD_4, self.domain_promptT_4, self.domain_promptV_4, self.domain_promptV_4]], dim=0)
        domain_prompt = [domain_prompt1, domain_prompt2, domain_prompt3, domain_prompt4]
        
        task_promptRD1 = torch.cat([self.task_promptR_1.repeat(3 * self.bs // 6,1,1), self.task_promptCR_1.repeat(self.bs // 6,1,1)], dim=0)
        task_promptRD2 = torch.cat([self.task_promptR_2.repeat(3 * self.bs // 6,1,1), self.task_promptCR_2.repeat(self.bs // 6,1,1)], dim=0)
        task_promptRD3 = torch.cat([self.task_promptR_3.repeat(3 * self.bs // 6,1,1), self.task_promptCR_3.repeat(self.bs // 6,1,1)], dim=0)
        task_promptRD4 = torch.cat([self.task_promptR_4.repeat(3 * self.bs // 6,1,1), self.task_promptCR_4.repeat(self.bs // 6,1,1)], dim=0)
        task_promptD = [task_promptRD1, task_promptRD2, task_promptRD3, task_promptRD4]

        _, _, depth_fea_1_16, depth_fea_1_32 = self.rgb_backbone(depth_Input[:-4, :, :, :], task_promptD, self.num_task, domain_prompt)

        depth_fea_1_32 = self.mlp32(depth_fea_1_32)
        depth_fea_1_16 = self.mlp16(depth_fea_1_16)
        depth_fea_1_16 = self.fuse_32_16(depth_fea_1_32, depth_fea_1_16)
        depth_fea_1_16 = self.mlp1(self.norm1(depth_fea_1_16))

        rgb_fea_1_32 = self.mlp32(rgb_fea_1_32)
        rgb_fea_1_16 = self.mlp16(rgb_fea_1_16)
        rgb_fea_1_16 = self.fuse_32_16(rgb_fea_1_32, rgb_fea_1_16)
        rgb_fea_1_16 = self.mlp1(self.norm1(rgb_fea_1_16))

        # Training
        rgb_fea = rgb_fea_1_16[-4:, :, :]  # 2, HW, C
        rgb_fea_1_16 = torch.cat([rgb_fea_1_16[:-4, :, :], depth_fea_1_16], dim=-1)
        rgb_fea_1_16 = self.mlp_s(self.norm(rgb_fea_1_16))
        rgb_fea_1_16 = torch.cat((rgb_fea_1_16, rgb_fea), dim=0)

        # VST Convertor
        rgb_fea_1_16 = self.transformer(rgb_fea_1_16)

        # VST Decoder
        fea_1_16, saliency_tokens, contour_tokens, fea_16, saliency_tokens_tmp, contour_tokens_tmp, saliency_fea_1_16, contour_fea_1_16, task_prompt, num1, task_prompt_total1, task_prompt1 = self.token_trans(rgb_fea_1_16)
        outputs_saliency, outputs_contour, outputs_saliency_s, outputs_contour_s, fea_fg, fea_bg = self.decoder(fea_1_16, saliency_tokens, contour_tokens, fea_16, saliency_tokens_tmp, contour_tokens_tmp, saliency_fea_1_16, contour_fea_1_16, rgb_fea_1_8, rgb_fea_1_4, task_prompt, num1)

        # for training
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

        return outputs_saliency, outputs_contour, outputs_saliency_s, outputs_contour_s, domain_prompt_total, task_prompt_total2, task_prompt_total1
