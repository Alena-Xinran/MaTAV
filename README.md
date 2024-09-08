# Mamba-Enhanced Text-Audio-Video Alignment Network for Emotion Recognition in Conversations

## Overview
MaTAV: A Mamba-enhanced Text-Audio-Video Alignment Network designed for Emotion Recognition in Conversations (ERC), aligning multimodal data to ensure consistency and capturing contextual emotional shifts in long dialogues, achieving superior performance on MELD and IEMOCAP datasets.

### Environment setup
```
# Environment: Python 3.8 + Torch 1.10.0 + CUDA 11.3
# Hardware: single RTX 3090 GPU, 48GB RAM
conda create --name MaTAV python=3.8
conda activate MaTAV
```
### Install dependencies
```
cd MaTAV
pip install -r requirements.txt
```

"Emotion Recognition in Conversations (ERCs) is a vital area within multimodal interaction research, dedicated to accurately identifying and classifying the emotions expressed by speakers throughout a conversation. Traditional ERC approaches predominantly rely on unimodal cues—such as text, audio, or visual data—leading to limitations in their effectiveness. These methods encounter two significant challenges: 1)Consistency in multimodal information. Before integrating various modalities, it is crucial to ensure that the data from different sources is aligned and coherent. 2)Contextual information capture. Successfully fusing multimodal features requires a keen understanding of the evolving emotional tone, especially in lengthy dialogues where emotions may shift and develop over time. To address these limitations, we propose a novel Mamba-enhanced Text-Audio-Video alignment network (MaTAV) for the ERC task. MaTAV is with the advantages of aligning unimodal features to ensure consistency across different modalities and handling long input sequences to better capture contextual multimodal information. The extensive experiments on the MELD and IEMOCAP datasets demonstrate that MaTAV significantly outperforms existing state-of-the-art methods on the ERC task with a big margin.",

