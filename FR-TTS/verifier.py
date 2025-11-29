
import torch
import torchvision.transforms as T
from typing import List
from PIL import Image
import numpy as np
from collections import defaultdict
import torch.nn as nn

from transformers import AutoImageProcessor,CLIPProcessor, CLIPModel
def get_verifier(verifier_name):
    if verifier_name == 'IR':
        import ImageReward as RM
        reward_model = RM.load('ImageReward-v1.0')
        return reward_model.score
    elif verifier_name == 'HPS':
        import hpsv2
        reward_model = hpsv2
        return reward_model.score
    elif verifier_name == 'AS':
        reward_model = AS_verifier()
        return reward_model.score
    elif verifier_name == 'CS':
        reward_model = ClipScorer()
        return reward_model
    elif verifier_name == 'GenEval':
        reward_model = GenEvalScorer()
        return reward_model.score
    elif verifier_name == 'HPSv3':
        import hpsv3
        reward_model = hpsv3.HPSv3RewardInferencer(config_path='/home/xuhang/code/HPSv3/HPSv3/config/HPSv3_7B.yaml',checkpoint_path='/home/xuhang/model/HPSv3/HPSv3.safetensors',device='cuda:1')
        # reward_model = hpsv3.HPSv3RewardInferencer(device='cuda:1')
        return reward_model.reward



class AS_verifier:
    def __init__(self):
        from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
        self.model, self.preprocessor = convert_v2_5_from_siglip(
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
        self.model = self.model.to(torch.bfloat16).cuda()
    
    def score(self,image):
        pixel_values = (
                self.preprocessor(images=image, return_tensors="pt")
                .pixel_values.to(torch.bfloat16)
                .cuda()
            )
            
        # Predict aesthetic score
        with torch.inference_mode():
            score = self.model(pixel_values).logits.squeeze().float().cpu().item()
        return score


def get_size(size):
    if isinstance(size, int):
        return (size, size)
    elif "height" in size and "width" in size:
        return (size["height"], size["width"])
    elif "shortest_edge" in size:
        return size["shortest_edge"]
    else:
        raise ValueError(f"Invalid size: {size}")
    
def get_image_transform(processor:AutoImageProcessor, include_to_tensor=True):
    config = processor.to_dict()
    resize = T.Resize(get_size(config.get("size"))) if config.get("do_resize") else nn.Identity()
    crop = T.CenterCrop(get_size(config.get("crop_size"))) if config.get("do_center_crop") else nn.Identity()
    normalise = T.Normalize(mean=processor.image_mean, std=processor.image_std) if config.get("do_normalize") else nn.Identity()

    transforms = [resize, crop]
    if include_to_tensor:
        transforms.append(T.ToTensor())  # 将 PIL.Image 转换为 tensor
    transforms.append(normalise)
    
    return T.Compose(transforms)

class ClipScorer(torch.nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device=device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        # 用于 PIL.Image 的 transform（包含 ToTensor）
        self.tform_pil = get_image_transform(self.processor.image_processor, include_to_tensor=True)
        # 用于 tensor 的 transform（不包含 ToTensor）
        self.tform_tensor = get_image_transform(self.processor.image_processor, include_to_tensor=False)
        self.eval()
    
    def _process(self, pixels):
        # 如果输入是列表，处理每个图像并堆叠成 batch
        if isinstance(pixels, list):
            processed = [self._process(img) for img in pixels]
            return torch.stack(processed)
        # 如果输入是 PIL.Image，使用包含 ToTensor 的 transform
        elif isinstance(pixels, Image.Image):
            pixels = self.tform_pil(pixels)
            # 确保有 batch 维度 [C, H, W] -> [1, C, H, W]
            if pixels.dim() == 3:
                pixels = pixels.unsqueeze(0)
            return pixels
        # 如果输入是 tensor，使用不包含 ToTensor 的 transform
        elif isinstance(pixels, torch.Tensor):
            dtype = pixels.dtype
            # 如果 tensor 是 3D [C, H, W]，先添加 batch 维度
            if pixels.dim() == 3:
                pixels = pixels.unsqueeze(0)
            # 应用 transform（可以处理 4D tensor [B, C, H, W]）
            pixels = self.tform_tensor(pixels)
            pixels = pixels.to(dtype=dtype)
            return pixels
        else:
            # 其他情况，尝试使用 PIL transform
            pixels = self.tform_pil(pixels)
            # 确保有 batch 维度
            if pixels.dim() == 3:
                pixels = pixels.unsqueeze(0)
            return pixels

    @torch.no_grad()
    def __call__(self, pixels, prompts, return_img_embedding=False):
        texts = self.processor(text=prompts, padding='max_length', truncation=True, return_tensors="pt").to(self.device)
        pixels = self._process(pixels).to(self.device)
        outputs = self.model(pixel_values=pixels, **texts)
        if return_img_embedding:
            return outputs.logits_per_image.diagonal()/30, outputs.image_embeds
        return outputs.logits_per_image.diagonal().cpu()/30

    @torch.no_grad()
    def image_similarity(self, pixels, ref_pixels):
        pixels = self._process(pixels).to(self.device)
        ref_pixels = self._process(ref_pixels).to(self.device)

        pixel_embeds = self.model.get_image_features(pixel_values=pixels)
        ref_embeds = self.model.get_image_features(pixel_values=ref_pixels)

        pixel_embeds = pixel_embeds / pixel_embeds.norm(p=2, dim=-1, keepdim=True)
        ref_embeds = ref_embeds / ref_embeds.norm(p=2, dim=-1, keepdim=True)

        sim = pixel_embeds @ ref_embeds.T
        sim = torch.diagonal(sim, 0)
        return sim


class GenEvalScorer:
    """Submits images to GenEval and computes a reward.
    """
    def __init__(self):
        import requests
        from requests.adapters import HTTPAdapter, Retry
        
        self.batch_size = 4
        self.url = "http://127.0.0.1:18085"
        self.sess = requests.Session()
        retries = Retry(
            total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
        )
        self.sess.mount("http://", HTTPAdapter(max_retries=retries))

    def score(self, images, prompts, metadatas, only_strict=False):
        import requests
        from requests.adapters import HTTPAdapter, Retry
        from io import BytesIO
        import pickle
        
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images_batched = np.array_split(images, np.ceil(len(images) / self.batch_size))
        metadatas_batched = np.array_split(metadatas, np.ceil(len(metadatas) / self.batch_size))
        all_scores = []
        all_rewards = []
        all_strict_rewards = []
        all_group_strict_rewards = []
        all_group_rewards = []
        for image_batch, metadata_batched in zip(images_batched, metadatas_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "meta_datas": list(metadata_batched),
                "only_strict": only_strict,
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = self.sess.post(self.url, data=data_bytes, timeout=120)
            response_data = pickle.loads(response.content)

            all_scores += response_data["scores"]
            all_rewards += response_data["rewards"]
            all_strict_rewards += response_data["strict_rewards"]
            all_group_strict_rewards.append(response_data["group_strict_rewards"])
            all_group_rewards.append(response_data["group_rewards"])
        all_group_strict_rewards_dict = defaultdict(list)
        all_group_rewards_dict = defaultdict(list)
        for current_dict in all_group_strict_rewards:
            for key, value in current_dict.items():
                all_group_strict_rewards_dict[key].extend(value)
        all_group_strict_rewards_dict = dict(all_group_strict_rewards_dict)

        for current_dict in all_group_rewards:
            for key, value in current_dict.items():
                all_group_rewards_dict[key].extend(value)
        all_group_rewards_dict = dict(all_group_rewards_dict)

        return all_rewards