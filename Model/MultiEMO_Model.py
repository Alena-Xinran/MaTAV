from DialogueRNN import BiModel
from MultiAttn import MultiAttnModel
from mamba import MambaModel
from MLP import MLP
import torch.nn.functional as F
import torch
import torch.nn as nn
class MultiEMO(nn.Module):
    
    def __init__(self, dataset, multi_attn_flag, roberta_dim, hidden_dim, dropout, num_layers, 
                 model_dim, num_heads, D_m_audio, D_m_visual, D_g, D_p, D_e, D_h,
                 n_classes, n_speakers, listener_state, context_attention, D_a, dropout_rec, device):
        super().__init__()
        self.device = device
        self.dataset = dataset
        self.multi_attn_flag = multi_attn_flag

        # Initialize feature transformation layers

        self.text_fc = nn.Linear(roberta_dim, model_dim).to(device)

        self.audio_fc = nn.Linear(D_m_audio, model_dim).to(device)
        
        self.visual_fc = nn.Linear(D_m_visual, model_dim).to(device)

        # Dialogue RNN for each modality

        self.text_dialoguernn = BiModel(model_dim, D_g, D_p, D_e, D_h, dataset, n_classes, n_speakers, listener_state, context_attention, D_a, dropout_rec, dropout, device)
        self.audio_dialoguernn = BiModel(model_dim, D_g, D_p, D_e, D_h, dataset, n_classes, n_speakers, listener_state, context_attention, D_a, dropout_rec, dropout, device)
        self.visual_dialoguernn = BiModel(model_dim, D_g, D_p, D_e, D_h, dataset, n_classes, n_speakers, listener_state, context_attention, D_a, dropout_rec, dropout, device)

        # Multimodal fusion
        # self.multiattn = MultiAttnModel(num_layers, model_dim, num_heads, hidden_dim, dropout)
        self.fc = nn.Linear(model_dim * 3, model_dim)
        self.mamba = MambaModel(d_model=model_dim, state_size=model_dim if dataset == 'IEMOCAP' else model_dim * 2, num_classes=n_classes, dropout_rate=dropout, device=device)
        if self.dataset == 'MELD':
            self.mlp = MLP(model_dim, model_dim * 2, n_classes, dropout)
        elif self.dataset == 'IEMOCAP':
            self.mlp = MLP(model_dim, model_dim, n_classes, dropout)
    def forward(self, texts, audios, visuals, speaker_masks, utterance_masks, padded_labels):
        # Assume texts are padded and you derive a mask from these
        # Creating a simple mask where non-zero entries are considered as data
        mask = (texts != 0).any(dim=-1)  # Adjust the dimension if necessary

        # Feature transformations and dialog context modeling
        text_features = self.text_dialoguernn(self.text_fc(texts), speaker_masks, utterance_masks)
        audio_features = self.audio_dialoguernn(self.audio_fc(audios), speaker_masks, utterance_masks)
        visual_features = self.visual_dialoguernn(self.visual_fc(visuals), speaker_masks, utterance_masks)
    

        # # 使用掩码来过滤数据，同时保持数据的三维结构
        # text_features = text_features[mask].view(-1, text_features.size(2))
        # audio_features = audio_features[mask].view(-1, audio_features.size(2))
        # visual_features = visual_features[mask].view(-1, visual_features.size(2))
        # print(text_features.shape)
        text_features = text_features.transpose(0, 1)
        audio_features = audio_features.transpose(0, 1)
        visual_features = visual_features.transpose(0, 1)
        # if self.multi_attn_flag == True:
        #     fused_text_features, fused_audio_features, fused_visual_features = self.multiattn(text_features, audio_features, visual_features)
        # else:
        #     fused_text_features, fused_audio_features, fused_visual_features = text_features, audio_features, visual_features
        # text_features = text_features[mask]
        # audio_features = audio_features[mask]
        # visual_features = visual_features[mask]
        fused_features = torch.cat((text_features, audio_features, visual_features), dim=2)
        fc_outputs = self.fc(fused_features)
        print("fc",fc_outputs.shape)
        #outputs1=fc_outputs.transpose(0, 1)
        outputs2 = fc_outputs.reshape(-1, fc_outputs.shape[-1])
        outputs2 = outputs2[padded_labels != -1]
        print("fc2",outputs2.shape)
        
        # Final emotion classification - now passing mask to MambaModel
        fc_outputs=fc_outputs.transpose(0, 1)
        #outputs = self.mamba(fc_outputs, mask)  # Ensure MambaModel's forward method uses this mask properly
        outputs = self.mlp(fc_outputs)
        outputs=outputs.transpose(0, 1)
        outputs = outputs.reshape(-1, outputs.shape[-1])
        outputs = outputs[padded_labels != -1]
        print("mamba",outputs.shape)
        fused_text_features = text_features.reshape(-1, text_features.shape[-1])
        fused_text_features = fused_text_features[padded_labels != -1]
        fused_audio_features = audio_features.reshape(-1, audio_features.shape[-1])
        fused_audio_features = fused_audio_features[padded_labels != -1]
        fused_visual_features = visual_features.reshape(-1, visual_features.shape[-1])
        fused_visual_features = fused_visual_features[padded_labels != -1]

        return fused_text_features, fused_audio_features, fused_visual_features, outputs2,outputs
