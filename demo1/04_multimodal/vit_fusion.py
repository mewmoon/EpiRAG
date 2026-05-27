import torch
import torch.nn as nn


class AnyResFeatureFusion(nn.Module):
    def __init__(self, feature_dim=1024, vit_grid_size=24):
        super().__init__()
        self.feature_dim = feature_dim
        self.vit_grid_size = vit_grid_size  # ViT输出的特征图宽高 (336/14 = 24)

        # 1. 模态对齐层 (Projection Layer)，将视觉维度映射到LLM维度
        self.llm_dim = 4096
        self.projector = nn.Linear(feature_dim, self.llm_dim)

        # 2. 定义可学习的 [Newline] 换行符 Token
        self.newline_token = nn.Parameter(torch.randn(1, 1, self.llm_dim))

    def forward(self, local_features, global_features):
        """
        Args:
            local_features: 4块局部切片的特征，形状为 (4, 576, 1024)
            global_features: 1块全局缩略图的特征，形状为 (1, 576, 1024)
        """
        B_local, seq_len, dim = local_features.shape  # 4, 576, 1024
        H_f = W_f = self.vit_grid_size  # 24, 24

        # ----------------------------------------------------
        # 步骤 1：将 4 块局部特征还原为 2x2 的空间二维网格
        # ----------------------------------------------------
        # 先把一维序列展回二维: (4, 576, 1024) -> (4, 24, 24, 1024)
        local_2d = local_features.view(4, H_f, W_f, dim)

        # 假设 4 块在空间上的顺序是: 左上、右上、左下、右下
        # 我们通过块切片拼接，拼成一个大的二维特征图 (48, 48, 1024)
        top_row = torch.cat(
            [local_2d[0], local_2d[1]], dim=1
        )  # 拼接宽度 W: (24, 48, 1024)
        bottom_row = torch.cat(
            [local_2d[2], local_2d[3]], dim=1
        )  # 拼接宽度 W: (24, 48, 1024)
        big_patch_2d = torch.cat(
            [top_row, bottom_row], dim=0
        )  # 拼接高度 H: (48, 48, 1024)

        current_H, current_W, _ = big_patch_2d.shape  # 48, 48

        # ----------------------------------------------------
        # 步骤 2：投影到 LLM 的特征维度
        # ----------------------------------------------------
        # (48, 48, 1024) -> (48, 48, 4196)
        big_patch_llm = self.projector(big_patch_2d)

        # ----------------------------------------------------
        # 步骤 3：精髓——在每一行的末尾插入 [Newline] Token
        # ----------------------------------------------------
        combined_rows = []
        # 准备一个广播形式的 Newline Token, 形状 (1, 4096)
        newline = self.newline_token.squeeze(0)

        for i in range(current_H):
            row = big_patch_llm[i]  # 提取第 i 行的特征: (48, 4096)
            # 在当前行的末尾拼接一个换行符: (48, 4096) + (1, 4096) -> (49, 4096)
            row_with_newline = torch.cat([row, newline], dim=0)
            combined_rows.append(row_with_newline)

        # 将所有带有换行符的行，级联成一个一维长序列
        # (48 行 * 49 个Token) = 2352 个视觉 Token
        flattened_local_tokens = torch.cat(combined_rows, dim=0)  # 形状: (2352, 4096)

        # ----------------------------------------------------
        # 步骤 4：融合全局缩略图特征
        # ----------------------------------------------------
        # 全局特征同样过 Projector: (1, 576, 1024) -> (576, 4096)
        global_llm = self.projector(global_features).squeeze(0)

        # 将全局低分辨特征与带换行符的高分辨局部特征拼接在一起
        # 形状: (576 + 2352, 4096) = (2928, 4096)
        final_visual_tokens = torch.cat([global_llm, flattened_local_tokens], dim=0)

        # 增加 Batch 维度返回 (Batch, Seq_Len, LLM_Dim)
        return final_visual_tokens.unsqueeze(0)


# --- 测试代码运行 ---
if __name__ == "__main__":
    # 模拟 ViT 产出的特征
    mock_local_feats = torch.randn(4, 576, 1024)  # 4块局部切片
    mock_global_feats = torch.randn(1, 576, 1024)  # 1块全局缩略图

    anyres_layer = AnyResFeatureFusion()
    output_tokens = anyres_layer(mock_local_feats, mock_global_feats)

    print("===== AnyRes 特征拼接测试 =====")
    print(f"输入局部特征形状: {mock_local_feats.shape}")
    print(f"输入全局特征形状: {mock_global_feats.shape}")
    print(f"最终喂给 LLM 的视觉 Token 形状: {output_tokens.shape}")
    # 期待输出: torch.Size([1, 2928, 4096])
