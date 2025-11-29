# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import torch
from transformers import AutoModelForCausalLM

from janus.models import MultiModalityCausalLM, VLChatProcessor
import numpy as np
import os
import PIL.Image
from verifier import get_verifier
from functools import partial
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import json
import math






def calculate_individual_diversity_scores(images):
    """
    计算每个样本与之前样本的最小VGG距离作为相似度，然后转换为多样性分数
    
    Args:
        images: List[PIL.Image], 图像列表
    
    Returns:
        diversity_scores: [len(images)] 每个样本的多样性分数列表
    """
    global global_vgg_model, global_vgg_transform
    
    if len(images) <= 1:
        return [1.0] if len(images) == 1 else []
    
    # 确保VGG模型已初始化
    if global_vgg_model is None or global_vgg_transform is None:
        init_vgg_model()
    
    # 预处理图像
    device = next(global_vgg_model.parameters()).device
    processed_images = []
    for img in images:
        # 确保图像是RGB格式
        if img.mode != 'RGB':
            img = img.convert('RGB')
        processed_img = global_vgg_transform(img).unsqueeze(0)
        processed_images.append(processed_img)
    
    # 将所有图像堆叠成一个batch
    batch_tensor = torch.cat(processed_images, dim=0).to(device)
    
    # 提取图像特征
    with torch.no_grad():
        # 使用VGG的features部分提取特征
        features = global_vgg_model.features(batch_tensor)  # [batch_size, 512, H, W]
        # 全局平均池化
        features = F.adaptive_avg_pool2d(features, (1, 1))  # [batch_size, 512, 1, 1]
        features = features.view(features.size(0), -1)  # [batch_size, 512]
        # 归一化特征
        image_features = F.normalize(features, p=2, dim=1)
    
    # 计算每个图像与之前图像的最小距离作为相似度
    diversity_scores = []
    for i in range(len(images)):
        if i == 0:
            # 第一个样本，没有之前的样本可比较，多样性分数为固定最大值1.0
            diversity_scores.append(1.0)
        else:
            # 计算与之前所有样本的最小欧式距离
            min_distance = float('inf')
            for j in range(i):
                dist = torch.norm(image_features[i] - image_features[j], p=2).item()
                min_distance = min(min_distance, dist)
            
            # 将最小距离作为相似度分数
            # 距离越小，相似度越高，多样性越低
            # 这里直接使用最小距离作为相似度分数
            diversity_scores.append(min_distance)
            
    diversity_scores[0] = max(diversity_scores[1:])
    return diversity_scores

def calculate_region_distance(region1, region2):
    """
    计算两个区域之间的欧式距离
    
    Args:
        region1: [token1, token2, token3] 第一个区域的token
        region2: [token1, token2, token3] 第二个区域的token
    
    Returns:
        distance: float 欧式距离
    """
    if len(region1) != len(region2):
        return float('inf')
    
    distance = 0
    for t1, t2 in zip(region1, region2):
        distance += (t1 - t2) ** 2
    
    return np.sqrt(distance)


def decode_and_get_rewards(generated_tokens, mmgpt, parallel_size, img_size, patch_size, 
                          current_step, total_steps, reward_model, original_prompt, decode_batch_size=8, 
                          use_multi_random_padding=True, random_trials=5, metadata=None, 
                          use_random_candidate_sampling=False):
    """
    分批解码生成的token并计算reward，避免显存溢出
    
    Args:
        generated_tokens: [parallel_size, image_token_num_per_image] 生成的token
        mmgpt: 模型
        parallel_size: 并行样本数量
        img_size: 图像尺寸
        patch_size: patch尺寸
        current_step: 当前步骤
        total_steps: 总步骤数
        reward_model: reward计算函数
        original_prompt: 原始prompt
        decode_batch_size: 解码时的批次大小
        use_multi_random_padding: 是否使用多次随机填充
        random_trials: 每个样本的随机填充次数
        metadata: 元数据信息
        use_random_candidate_sampling: 是否使用随机候选采样方法
    
    Returns:
        rewards: [parallel_size] reward列表
        all_images: [parallel_size] 所有解码的图像列表（统一返回裁剪后的图像）
    """
    all_rewards = []
    all_images = []
    
    if use_random_candidate_sampling:
        # 使用新的随机候选采样方法：先随机填充一次，然后迭代优化
        for sample_idx in range(parallel_size):
            # 为当前样本填充未生成的部分
            filled_tokens = generated_tokens[sample_idx:sample_idx+1].clone()
            
            # 计算已生成的token数量和总token数量
            generated_token_count = current_step
            total_token_count = generated_tokens.shape[1]
            generated_blocks = generated_token_count // block_size
            total_blocks = total_token_count // block_size
            
            if generated_blocks == 0:
                # 如果没有任何已生成的块，使用原始填充方法
                all_rewards.append(0.0)  # 默认reward
                all_images.append(PIL.Image.new('RGB', (img_size, img_size), (0, 0, 0)))
                continue
            
            # 第一步：生成init_candidate_size个初始候选方案
            initial_candidates = []
            for init_idx in range(init_candidate_size):
                candidate = filled_tokens.clone()
                for block_idx in range(generated_blocks, total_blocks):
                    random_block_idx = np.random.randint(0, generated_blocks)
                    source_start = random_block_idx * block_size
                    source_end = (random_block_idx + 1) * block_size
                    target_start = block_idx * block_size
                    target_end = min((block_idx + 1) * block_size, total_token_count)
                    candidate[0, target_start:target_end] = \
                        generated_tokens[sample_idx, source_start:source_end]
                initial_candidates.append(candidate)
            
            # 计算所有初始候选方案的reward
            initial_rewards = []
            initial_images = []
            for candidate in initial_candidates:
                dec = mmgpt.gen_vision_model.decode_code(
                    candidate.to(dtype=torch.int), 
                    shape=[1, 8, img_size//patch_size, img_size//patch_size]
                )
                dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
                dec = np.clip((dec + 1) / 2 * 255, 0, 255)
                
                full_visual_img = np.zeros((1, img_size, img_size, 3), dtype=np.uint8)
                full_visual_img[0, :, :] = dec[0]
                
                visual_img = PIL.Image.fromarray(full_visual_img[0])
                reward = reward_model([visual_img], original_prompt)
                if not isinstance(reward, list):
                    reward = [reward]
                reward_value = np.max(reward)
                
                initial_rewards.append(reward_value)
                initial_images.append(visual_img)
            
            # 选择reward最高的作为初始方案
            best_init_idx = np.argmax(initial_rewards)
            best_reward = initial_rewards[best_init_idx]
            best_candidate = initial_candidates[best_init_idx].clone()
            best_image = initial_images[best_init_idx]
            
            # 迭代优化：每次随机两次替换，如果有更好的就保留
            no_improvement_count = 0
            iteration_count = 0
            
            while iteration_count < candidate_size and no_improvement_count < 2:
                iteration_count += 1
                improved = False
                
                # 每次迭代尝试两次随机替换
                for attempt in range(2):
                    # 创建新的候选方案
                    new_candidate = best_candidate.clone()
                    
                    # 随机选择25%的未生成块进行重新填充
                    ungenerated_blocks = list(range(generated_blocks, total_blocks))
                    if len(ungenerated_blocks) > 0:
                        # 随机选择25%的块
                        num_blocks_to_replace = max(1, len(ungenerated_blocks) // 4)
                        blocks_to_replace = np.random.choice(
                            ungenerated_blocks, 
                            size=min(num_blocks_to_replace, len(ungenerated_blocks)), 
                            replace=False
                        )
                        
                        # 重新填充选中的块
                        for block_idx in blocks_to_replace:
                            random_block_idx = np.random.randint(0, generated_blocks)
                            source_start = random_block_idx * block_size
                            source_end = (random_block_idx + 1) * block_size
                            target_start = block_idx * block_size
                            target_end = min((block_idx + 1) * block_size, total_token_count)
                            new_candidate[0, target_start:target_end] = \
                                generated_tokens[sample_idx, source_start:source_end]
                    
                    # 计算新方案的reward
                    dec = mmgpt.gen_vision_model.decode_code(
                        new_candidate.to(dtype=torch.int), 
                        shape=[1, 8, img_size//patch_size, img_size//patch_size]
                    )
                    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
                    dec = np.clip((dec + 1) / 2 * 255, 0, 255)
                    
                    full_visual_img = np.zeros((1, img_size, img_size, 3), dtype=np.uint8)
                    full_visual_img[0, :, :] = dec[0]
                    
                    visual_img = PIL.Image.fromarray(full_visual_img[0])
                    new_reward = reward_model([visual_img], original_prompt)
                    if not isinstance(new_reward, list):
                        new_reward = [new_reward]
                    new_reward = np.max(new_reward)
                    
                    # 如果新方案更好，则保留并跳出内层循环
                    if new_reward > best_reward:
                        best_reward = new_reward
                        best_candidate = new_candidate.clone()
                        best_image = visual_img
                        improved = True
                        break  # 跳出两次尝试的循环
                
                # 更新改进计数器
                if improved:
                    no_improvement_count = 0  # 重置计数器
                else:
                    no_improvement_count += 1
            
            # 返回最佳方案
            all_rewards.append(best_reward)
            all_images.append(best_image.crop((0, 0, img_size, current_step // (img_size//patch_size) * patch_size)))

    elif use_multi_random_padding:
        # 对每个样本进行多次随机填充，取最高reward
        for sample_idx in range(parallel_size):
            sample_reward = []
            # 进行多次随机填充试验
            for trial in range(random_trials):
                # 为当前样本填充未生成的部分
                filled_tokens = generated_tokens[sample_idx:sample_idx+1].clone()
                
                # 按 block_size 的token块进行填充
                generated_token_count = current_step
                total_token_count = generated_tokens.shape[1]
                generated_blocks = generated_token_count // block_size
                total_blocks = total_token_count // block_size
                for block_idx in range(generated_blocks, total_blocks):
                    if generated_blocks > 0:
                        random_block_idx = np.random.randint(0, generated_blocks)
                        source_start = random_block_idx * block_size
                        source_end = (random_block_idx + 1) * block_size
                        target_start = block_idx * block_size
                        target_end = min((block_idx + 1) * block_size, total_token_count)
                        filled_tokens[0, target_start:target_end] = \
                            generated_tokens[sample_idx, source_start:source_end]
                
                # 解码当前样本
                dec = mmgpt.gen_vision_model.decode_code(
                    filled_tokens.to(dtype=torch.int), 
                    shape=[1, 8, img_size//patch_size, img_size//patch_size]
                )
                dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
                dec = np.clip((dec + 1) / 2 * 255, 0, 255)
                
                # 创建完整的可视化图像（用于reward计算）
                full_visual_img = np.zeros((1, img_size, img_size, 3), dtype=np.uint8)
                full_visual_img[0, :, :] = dec[0]
                
                # 转换为PIL图像
                visual_img_full = PIL.Image.fromarray(full_visual_img[0])
                # os.makedirs('./test_img',exist_ok=True)
                # visual_img_full.save(f'./test_img/{generated_blocks}_{sample_idx}_{trial}.png')
                
                # 计算reward（使用完整图像）
                trial_reward = reward_model([visual_img_full], original_prompt)
                if not isinstance(trial_reward, list):
                    trial_reward = [trial_reward]
                sample_reward.append(trial_reward)
            
            all_rewards.append(np.max(sample_reward))
            all_images.append(visual_img_full.crop((0, 0, img_size, current_step // (img_size//patch_size) * patch_size)))

    
    else:
        # 直接解码，不进行随机填充
        # 分批解码
        for start_idx in range(0, parallel_size, decode_batch_size):
            end_idx = min(start_idx + decode_batch_size, parallel_size)
            batch_tokens = generated_tokens[start_idx:end_idx]
            batch_size = end_idx - start_idx
            
            # 解码当前批次的token
            dec = mmgpt.gen_vision_model.decode_code(
                batch_tokens.to(dtype=torch.int), 
                shape=[batch_size, 8, img_size//patch_size, img_size//patch_size]
            )
            dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
            dec = np.clip((dec + 1) / 2 * 255, 0, 255)
            
            # 创建完整的可视化图像（用于reward计算）
            full_visual_img = np.zeros((batch_size, img_size, img_size, 3), dtype=np.uint8)
            full_visual_img[:, :, :] = dec
            
            # 转换为PIL图像列表
            visual_imgs_full = [PIL.Image.fromarray(img) for img in full_visual_img]
            
            # 计算reward（根据only_generated_image决定使用完整图像还是裁剪图像）
            if use_crop_resize_reward:
                # 对已生成部分进行裁剪并resize到384*384
                current_height = current_step // (img_size//patch_size) * patch_size
                cropped_imgs = [img.crop((0, 0, img_size,current_height)) for img in visual_imgs_full]
                resized_imgs = [img.resize((img_size,img_size), PIL.Image.LANCZOS) for img in cropped_imgs]
                batch_rewards = reward_model(resized_imgs, original_prompt)
            elif only_generated_image:
                # 使用裁剪后的图像计算reward
                current_height = current_step // (img_size//patch_size) * patch_size
                cropped_imgs = [img.crop((0, 0, img_size, current_height)) for img in visual_imgs_full]
                batch_rewards = reward_model(cropped_imgs, original_prompt)
            else:
                # 使用完整图像计算reward
                batch_rewards = reward_model(visual_imgs_full, original_prompt)
            
            if not isinstance(batch_rewards, list):
                batch_rewards = [batch_rewards]
            
            # 统一返回裁剪后的图像
            current_height = current_step // (img_size//patch_size) * patch_size
            cropped_imgs = [img.crop((0, 0, img_size, current_height)) for img in visual_imgs_full]
            
            all_images.extend(cropped_imgs)
            all_rewards.extend(batch_rewards)
    
    return all_rewards, all_images





# for transformers==4.46
# def prune_by_indices(
#     generated_tokens: torch.Tensor,
#     outputs,
#     probs,
#     confidence,
#     keep_indices,
# ):
#     device = generated_tokens.device
#     # 只创建一次keep_idx_tensor，避免重复创建
#     keep_idx_tensor = torch.tensor(keep_indices, device=device, dtype=torch.long)
    
#     # 就地选择tokens，减少内存使用
#     new_generated_tokens = generated_tokens.index_select(0, keep_idx_tensor)
    
#     # 计算idx_pair一次，避免重复计算
#     idx_pair = torch.stack([2 * keep_idx_tensor, 2 * keep_idx_tensor + 1], dim=1).reshape(-1)
    
#     # 就地更新past_key_values，减少内存使用
#     new_past_key_values = []
#     for layer_idx in range(len(outputs.past_key_values)):
#         k_layer, v_layer = outputs.past_key_values[layer_idx]
#         # 就地选择，避免创建新的tensor
#         new_k_layer = k_layer.index_select(0, idx_pair)
#         new_v_layer = v_layer.index_select(0, idx_pair)
#         new_past_key_values.append((new_k_layer, new_v_layer))
        
#         # 显式删除原始tensor以释放内存
#         del k_layer, v_layer
    
#     # 更新outputs并删除旧的past_key_values
#     old_past_key_values = outputs.past_key_values
#     outputs.past_key_values = tuple(new_past_key_values)
#     del old_past_key_values, new_past_key_values
    
#     # 就地选择probs
#     new_probs = probs.index_select(0, keep_idx_tensor)
#     new_confidence = confidence.index_select(0, keep_idx_tensor) 
    
#     # 显式删除不需要的tensor
#     del keep_idx_tensor, idx_pair
    

#     return new_generated_tokens, outputs, new_probs, new_confidence


#for transformers==4.57.0
def prune_by_indices(
    generated_tokens: torch.Tensor,
    outputs,
    probs,
    confidence,
    kl_seq,
    keep_indices,
):
    device = generated_tokens.device
    # 只创建一次keep_idx_tensor，避免重复创建
    keep_idx_tensor = torch.tensor(keep_indices, device=device, dtype=torch.long)
    
    # 就地选择tokens，减少内存使用
    new_generated_tokens = generated_tokens.index_select(0, keep_idx_tensor)
    
    # 计算idx_pair一次，避免重复计算
    idx_pair = torch.stack([2 * keep_idx_tensor, 2 * keep_idx_tensor + 1], dim=1).reshape(-1)
    
    outputs.past_key_values.batch_select_indices(idx_pair)
    
    # 就地选择probs
    new_probs = probs.index_select(0, keep_idx_tensor)
    new_confidence = confidence.index_select(0, keep_idx_tensor)
    new_kl_seq = kl_seq.index_select(0, keep_idx_tensor)
    # 显式删除不需要的tensor
    del keep_idx_tensor, idx_pair
    

    return new_generated_tokens, outputs, new_probs, new_confidence, new_kl_seq




def score_model(images,prompt,verifier_name,verifier,metadata=None):
    if verifier_name == 'IR':
        return verifier(prompt,images)
    elif verifier_name == 'HPS':
        return verifier(images,prompt)
    elif verifier_name == 'AS':
        def verifier_batch(images,prompt):
            score_list = [verifier(image) for image in images]
            return score_list
        return verifier_batch(images,prompt)
    elif verifier_name == 'CS':
        def verifier_batch(images,prompt):
            score_list = [verifier(image,prompt) for image in images]
            return score_list
        return verifier_batch(images,prompt)
    elif verifier_name == 'GenEval':
        return verifier(images,prompt,metadata,False)
    elif verifier_name == 'HPSv3':
        reward_list = verifier(images,[prompt]*len(images))
        reward_list = [reward[0].item() for reward in reward_list]
        return reward_list
    
def init_random(random_seed):
    np.random.seed(int(random_seed))
    torch.manual_seed(int(random_seed))
    torch.cuda.manual_seed(int(random_seed))
    # generator = torch.manual_seed(random_seed)  # 暂时不使用

def clear_gpu_cache():
    """清理GPU缓存以释放显存"""
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def init_vgg_model(vgg_model_name="vgg16"):
    """初始化全局VGG模型"""
    global global_vgg_model, global_vgg_transform
    
    if global_vgg_model is None or global_vgg_transform is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载预训练的VGG模型
        if vgg_model_name == "vgg16":
            global_vgg_model = models.vgg16(pretrained=True).to(device)
        elif vgg_model_name == "vgg19":
            global_vgg_model = models.vgg19(pretrained=True).to(device)
        else:
            raise ValueError(f"Unsupported VGG model: {vgg_model_name}")
        
        global_vgg_model.eval()
        
        # 定义图像预处理
        global_vgg_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # VGG需要224x224的输入
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet标准化
        ])
    
    return global_vgg_model, global_vgg_transform


def clear_vgg_model():
    """清理VGG模型以释放显存"""
    global global_vgg_model, global_vgg_transform
    
    if global_vgg_model is not None:
        del global_vgg_model
        global_vgg_model = None
    
    if global_vgg_transform is not None:
        del global_vgg_transform
        global_vgg_transform = None
    
    clear_gpu_cache()

def get_dynamic_weights(current_step, total_steps, img_size, patch_size, rewards_variance=None):
    """
    根据当前步骤和rewards方差计算动态权重
    
    Args:
        current_step: 当前步骤
        total_steps: 总步骤数
        img_size: 图像尺寸
        patch_size: patch尺寸
        rewards_variance: rewards的方差，用于自适应调整权重
    
    Returns:
        diversity_weight, reward_weight: 当前步骤的权重
    """
    if not use_dynamic_weights:
        return late_diversity_weight, 1.0 - late_diversity_weight
    
    # 计算当前是第几行（从0开始）
    current_row = current_step // (img_size // patch_size)
    total_rows = total_steps // (img_size // patch_size)
    
    if total_rows <= 1:
        return early_diversity_weight, 1.0 - early_diversity_weight
    
    # 确定多样性权重变化的结束行
    end_row = diversity_end_row if diversity_end_row is not None else total_rows - 1
    
    # 根据初始行和结束行计算基础权重
    if current_row < diversity_start_row:
        # 初始行之前，多样性权重为1
        base_diversity_weight = 1.0
        base_reward_weight = 0.0
    elif current_row > end_row:
        # 结束行之后，多样性权重为0
        base_diversity_weight = 0.0
        base_reward_weight = 1.0
    else:
        # 在初始行和结束行之间，进行线性插值
        if end_row > diversity_start_row:
            progress = (current_row - diversity_start_row) / (end_row - diversity_start_row)
            progress = max(0, min(1, progress))  # 确保在[0, 1]范围内
        else:
            progress = 0.0
        
        base_diversity_weight = early_diversity_weight * (1 - progress) + late_diversity_weight * progress
        base_reward_weight = 1.0 - base_diversity_weight
    
    # 如果有rewards方差信息且启用方差自适应，进行自适应调整
    if rewards_variance is not None and use_variance_adaptation:
        # 归一化方差到[0, 1]范围
        # 使用sigmoid函数将方差映射到[0, 1]
        rewards_variance = rewards_variance - variance_center
        normalized_variance = 1 / (1 + np.exp(-rewards_variance * variance_sensitivity))
        
        # 方差大时增加reward权重，方差小时增加diversity权重
        # 调整幅度在[-adjustment_range, adjustment_range]之间
        variance_adjustment = (normalized_variance - 0.5) * 2 * adjustment_range
        
        # 应用调整
        current_diversity_weight = base_diversity_weight - variance_adjustment
        current_reward_weight = base_reward_weight + variance_adjustment
        
        # 确保权重在合理范围内
        current_diversity_weight = max(0.0, min(1.0, current_diversity_weight))
        current_reward_weight = max(0.0, min(1.0, current_reward_weight))
    else:
        # 没有方差信息或未启用方差自适应时使用基础权重
        current_diversity_weight = base_diversity_weight
        current_reward_weight = base_reward_weight
    
    return current_diversity_weight, current_reward_weight


def importance_sampling_selection(combined_scores, target_size, temperature=1.0, top_k=None):
    """
    基于重要性采样的样本选择，使用softmax将分数转换为概率分布
    
    Args:
        combined_scores: 组合分数数组 [num_samples]
        target_size: 目标样本数量
        temperature: softmax温度参数，控制分布的尖锐程度
        top_k: 只在前top_k个高分样本中进行重要性采样，如果为None则使用所有样本
    
    Returns:
        selected_indices: 选中的样本索引（可能有重复）
    """
    # 如果指定了top_k，先选择前top_k个高分样本
    if top_k is not None and top_k < len(combined_scores):
        # 获取前top_k个高分样本的索引
        top_k_indices = np.argsort(combined_scores)[-top_k:]
        # 只在这些样本中进行重要性采样
        top_k_scores = [combined_scores[i] for i in top_k_indices]
        scores_tensor = torch.tensor(top_k_scores, dtype=torch.float32)
        probs = torch.softmax(scores_tensor / temperature, dim=0)
        
        # 重要性采样：根据概率分布进行有放回采样
        selected_top_k_indices = torch.multinomial(probs, num_samples=target_size, replacement=True)
        
        # 将top_k内的索引映射回原始索引
        selected_indices = [top_k_indices[idx.item()] for idx in selected_top_k_indices]
    else:
        # 使用所有样本进行重要性采样
        scores_tensor = torch.tensor(combined_scores, dtype=torch.float32)
        probs = torch.softmax(scores_tensor / temperature, dim=0)
        
        # 重要性采样：根据概率分布进行有放回采样
        selected_indices = torch.multinomial(probs, num_samples=target_size, replacement=True)
        selected_indices = selected_indices.tolist()
    
    return selected_indices



# load_model
model_path = "/home/xuhang/model/Janus-Pro-7B"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer
vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()


# load prompts
with open("./prompts_open_image_pref_v1.txt", "r", encoding="utf-8") as f:
    prompt_list = [line.strip() for line in f if line.strip()]
prompt_list = prompt_list[50:150]

# with open("/home/xuhang/code/Janus/geneval.jsonl", 'r', encoding='utf-8') as f:
#     metadata_list = [json.loads(line) for line in f]
#     prompt_list = [item['prompt'] for item in metadata_list]

# metadata_list = metadata_list[:]
# prompt_list = prompt_list[:]

processed_prompt_list = []
for prompt in prompt_list:  
    conversation = [
        {
            "role": "User",
            "content": prompt,
        },
        {"role": "Assistant", "content": ""},
    ]

    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    processed_prompt = sft_format + vl_chat_processor.image_start_tag
    processed_prompt_list.append(processed_prompt)

#load_parameters
# save_dir = "./output_demo/tts_IR_8_intermediate_select_multi_random_row_padding_every_12_tokens_trial_10_dynamic_weights_6_18_12row_4"
save_dir = "./output_7B_demo/tts_HPS_1_bon"
# save_dir = "./output_7B_demo/tts_IR_8_intermediate_select_multi_random_row_padding_every_24_tokens_trial_10_dynamic_weights_6_18_12row_4"
verifier_name = "HPS"
parallel_size = 1
temperature = 1
cfg_weight = 5
num_samples = 1
decode_batch_size = 4  # 解码时的批次大小，避免显存溢出
intermediate_select = False # 是否在中间步骤进行筛选

# 分裂与筛选超参数
max_branches_per_step = 1  # 每步分裂后的最大样本数预算
prune_keep_ratio = 0.25  # 在筛选集合中按 reward 仅保留前比例  # 是否使用内存优化版本的prune_by_indices（显存不足时设为True）

# 多样性计算超参数
use_diversity_screening = True  # 是否使用多样性计算进行筛选（False时仅使用reward）
vgg_model_name = "vgg16"  # VGG模型名称，用于多样性计算

# 动态权重变化参数C
early_diversity_weight = 1.0 # 前期多样性权重（第一行）
late_diversity_weight = 0.0   # 后期多样性权重（最后一行）
use_dynamic_weights = True    # 是否使用动态权重变化
diversity_start_row = 6     # 多样性权重开始变化的初始行（之前权重为1）
diversity_end_row = 18   # 多样性权重结束变化的结束行（之后权重为0），None表示使用总行数
filter_start_row = 12     # 筛选开始变化的初始行（之前权重为1）
interval_row = 6      # 筛选变化间隔的行数，None表示使用总行数

# 方差自适应参数
use_variance_adaptation = True  # 是否使用rewards方差进行自适应调整
variance_sensitivity = 10.0     # 方差敏感度参数，越大越敏感
adjustment_range = 0.3          # 权重调整范围，最大调整幅度
only_generated_image = False    # 是否只显示已生成的部分
use_crop_resize_reward = False    # 是否对已生成部分裁剪并resize到384*384后计算reward
variance_center = 0.03
# 随机候选采样参数
use_random_candidate_sampling = True # 是否使用随机候选采样方法
candidate_size=5
init_candidate_size=5  # 初始候选方案数量

# 重要性采样参数
importance_sampling_temperature = 1.0  # softmax温度，控制分布尖锐程度
use_importance_sampling = True         # 是否使用重要性采样替代固定比例筛选
importance_sampling_top_k = 16         # 只在前top_k个高分样本中进行重要性采样，None表示使用所有样本

# 随机填充参数
use_multi_random_padding = False # 是否使用多次随机填充
random_trials = 10  # 每个样本的随机填充次数，用于探索不同的行组合


block_size = 12



    

#load_verifier
verifier = get_verifier(verifier_name)
reward_model = partial(score_model,verifier_name=verifier_name,verifier=verifier)

verifier_2 = get_verifier('GenEval')
reward_model_2 = partial(score_model,verifier_name='GenEval',verifier=verifier_2)

# 全局VGG模型（避免重复加载）
global_vgg_model = None
global_vgg_transform = None

# 初始化VGG模型
# init_vgg_model(vgg_model_name)
def top_k_top_p_filtering(
    logits,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

def entropy_from_logits(logits):
    p = torch.softmax(logits, dim=-1)
    ent = -(p * (p + 1e-12).log()).sum(dim=-1)
    return ent


def top2_margin(logits):
    p = torch.softmax(logits, dim=-1)
    topk = torch.topk(p, k=2, dim=-1)[0]
    if topk.shape[-1] < 2:
        return torch.tensor(1.0, device=logits.device)
    return (topk[..., 0] - topk[..., 1])


def kl_between_logits(logits_p, logits_q):
    p = torch.softmax(logits_p, dim=-1)
    q = torch.softmax(logits_q, dim=-1)

    kl_per_position = (p * ((p + 1e-12).log() - (q + 1e-12).log())).sum(dim=-1)
    
    if kl_per_position.dim() > 1:
        return kl_per_position[:, -1] 
    return kl_per_position

@torch.inference_mode()
def generate(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    original_prompt:str,
    temperature: float = 1,
    parallel_size: int = 1,
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
    num_samples: int = 1,
):

    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).cuda()
    for i in range(parallel_size*2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()
    confidence= torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.float32).cuda()
    kl_seq = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.float32).cuda()


    outputs = None
    running_K_min = float('inf')
    running_K_max = float('-inf')
    for i in tqdm(range(image_token_num_per_image)):

        past_kv = outputs.past_key_values if i != 0 else None

        outputs = mmgpt.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=past_kv
        )

        hidden_states = outputs.last_hidden_state
        
        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        
        logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
        # logits = top_k_top_p_filtering(logits, top_k=1000, top_p=1.0)

        H_t = entropy_from_logits(logits)
        m_t = top2_margin(logits)
        S_t_raw = 0.5 * (H_t / (math.log(16384) + 1e-12)) + 0.5 * (1.0 - m_t)
        S_t = 1-S_t_raw
        if i == 0:
            confidence[:, i] = S_t
        else:
            confidence[:,i] = 0.8*confidence[:,i-1]+0.2*S_t
        
        K_t = kl_between_logits(logit_cond, logit_uncond).detach().cpu()
        running_K_min = min(running_K_min, K_t.min().item())
        running_K_max = max(running_K_max, K_t.max().item())

        if running_K_max > running_K_min:
            D_t = torch.clamp((K_t - running_K_min) / (running_K_max - running_K_min + 1e-12), 0.0, 1.0)
        else:
            D_t = torch.zeros_like(K_t)
        
        # 将D_t移动到与kl_seq相同的设备
        D_t = D_t.to(device=kl_seq.device, dtype=kl_seq.dtype)
        
        if i==0:
            kl_seq[:, i] = D_t
        else:
            kl_seq[:, i] = 0.8*kl_seq[:, i-1]+0.2*D_t
        

        probs = torch.softmax(logits/temperature, dim=-1)



        # 每行结束时进行筛选和分裂
        current_size_before_split = generated_tokens.shape[0]
        if intermediate_select  and i >=filter_start_row*(img_size // patch_size) and i % (interval_row*(img_size // patch_size)) == 0:
            # 1. 计算所有样本的rewards和解码图像
            rewards_current, decoded_images = decode_and_get_rewards(
                generated_tokens, mmgpt, current_size_before_split, img_size, patch_size,
                i, image_token_num_per_image, reward_model, original_prompt, decode_batch_size, 
                use_multi_random_padding, random_trials, metadata=None, 
                use_random_candidate_sampling=use_random_candidate_sampling
            )   
            rewards_current = np.array(rewards_current)
            
            # 计算rewards的方差
            rewards_variance = np.var(rewards_current)
            
            # 获取动态权重（基于rewards方差自适应调整）
            current_diversity_weight, current_reward_weight = get_dynamic_weights(
                i, image_token_num_per_image, img_size, patch_size, rewards_variance
            )
            
            if use_diversity_screening:
                # 2. 计算所有样本的多样性分数（基于图像CLIP相似度）
                diversity_scores = calculate_individual_diversity_scores(decoded_images)
                diversity_scores = np.array(diversity_scores)
                
                # 3. 标准化分数到[0, 1]范围
                if rewards_current.std() > 0:
                    rewards_normalized = (rewards_current - rewards_current.min()) / (rewards_current.max() - rewards_current.min())
                else:
                    rewards_normalized = np.ones_like(rewards_current)
                
                if diversity_scores.std() > 0:
                    diversity_normalized = (diversity_scores - diversity_scores.min()) / (diversity_scores.max() - diversity_scores.min())
                else:
                    diversity_normalized = np.ones_like(diversity_scores)
                
                # 4. 使用动态权重组合分数
                combined_scores = current_reward_weight * rewards_normalized + current_diversity_weight * diversity_normalized
                
            else:
                # 仅使用reward
                if rewards_current.std() > 0:
                    combined_scores = (rewards_current - rewards_current.min()) / (rewards_current.max() - rewards_current.min())
                else:
                    combined_scores = np.ones_like(rewards_current)
            
            # 5. 基于重要性采样的样本选择
            if use_importance_sampling:
                # 使用重要性采样选择样本到目标数量
                target_size = max_branches_per_step
                if i < 200:
                    importance_sampling_top_k = target_size
                elif i < 400:
                    importance_sampling_top_k = target_size // 2
                else:
                    importance_sampling_top_k = target_size // 4
                selected_indices = importance_sampling_selection(
                    combined_scores, target_size, importance_sampling_temperature, importance_sampling_top_k
                )
                
                
                generated_tokens, outputs, probs, confidence, kl_seq = prune_by_indices(
                    generated_tokens, outputs, probs, confidence, kl_seq, selected_indices
                )

                
            else:
                # 原始固定比例筛选逻辑
                keep_num = max(1, int(current_size_before_split * prune_keep_ratio))
                topk_indices = np.argsort(combined_scores)[-keep_num:]  # 选择分数最高的样本
                keep_indices_pre = sorted(topk_indices.tolist())
                
                # 扩展到target_size，通过重复高分样本
                target_size = max_branches_per_step
                if len(keep_indices_pre) < target_size:
                    # 计算需要重复的次数
                    repeat_times = target_size // len(keep_indices_pre)
                    remainder = target_size % len(keep_indices_pre)
                    
                    # 重复高分样本
                    expanded_indices = keep_indices_pre * repeat_times
                    
                    # 添加剩余的样本（从高分到低分）
                    if remainder > 0:
                        expanded_indices.extend(keep_indices_pre[:remainder])
                    
                    keep_indices_pre = expanded_indices
                elif len(keep_indices_pre) > target_size:
                    # 如果超过target_size，只保留前target_size个
                    keep_indices_pre = keep_indices_pre[:target_size]
                
                generated_tokens, outputs, probs, confidence, kl_seq = prune_by_indices(
                    generated_tokens, outputs, probs, confidence, kl_seq, selected_indices
                )
                
        
        # 常规步骤：采样下一个token（不进行分裂）
        # current_size = generated_tokens.shape[0]  # 暂时不使用
        
        # 每个样本采样1个 token，直接写入，不改变 batch 大小
        next_token_single = torch.multinomial(probs, num_samples=1)  # [B,1]
        next_token_idx = next_token_single.squeeze(-1)  # [B]
        generated_tokens[:, i] = next_token_idx

        # 准备下一步的图像 token embeds（cond/uncond 对齐）
        next_token_pair = torch.cat([next_token_idx.unsqueeze(1), next_token_idx.unsqueeze(1)], dim=1).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token_pair)
        inputs_embeds = img_embeds.unsqueeze(dim=1)
        


            
        
            




    # 使用当前的parallel_size进行最终解码（分批处理避免显存溢出）
    current_parallel_size = generated_tokens.shape[0]
    all_visual_imgs = []
    
    for start_idx in range(0, current_parallel_size, decode_batch_size):
        end_idx = min(start_idx + decode_batch_size, current_parallel_size)
        batch_tokens = generated_tokens[start_idx:end_idx]
        batch_size = end_idx - start_idx
        
        dec = mmgpt.gen_vision_model.decode_code(
            batch_tokens.to(dtype=torch.int), 
            shape=[batch_size, 8, img_size//patch_size, img_size//patch_size]
        )
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        dec = np.clip((dec + 1) / 2 * 255, 0, 255)

        visual_img = np.zeros((batch_size, img_size, img_size, 3), dtype=np.uint8)
        visual_img[:, :, :] = dec
        #numpy2pil
        batch_visual_imgs = [PIL.Image.fromarray(img) for img in visual_img]
        all_visual_imgs.extend(batch_visual_imgs)

    clear_gpu_cache()
    del outputs, inputs_embeds
    return all_visual_imgs, confidence, kl_seq



if __name__ == '__main__':
    # for a in [1,6,12,24,48]:
    #     for b in [1,5]:
    #         for c in [1]:
    #             block_size = a
    #             random_trials = b
    #             interval_row = c
    #             save_dir = f"./output/tts_{verifier_name}_{parallel_size}_intermediate_select_multi_random_row_padding_every_{block_size}_tokens_trial_{random_trials}_dynamic_weights_6_18_12row_{interval_row}"  
    score = 0
    geneval_reward = 0
    os.makedirs(save_dir,exist_ok=True)
    with torch.inference_mode():
        for idx, (prompt, processed_prompt) in enumerate(zip(prompt_list, processed_prompt_list)):
            # 为每个样本设置固定的随机种子以确保结果可复现
            init_random(idx)

            visual_imgs,confidence,kl_seq = generate(
                vl_gpt,
                vl_chat_processor,
                processed_prompt,
                prompt,
                temperature=temperature,
                parallel_size=parallel_size,
                cfg_weight=cfg_weight,
                num_samples=num_samples,
            )
            # 分批评估reward，避免显存溢出
            all_rewards = []
            for start_idx in range(0, len(visual_imgs), decode_batch_size):
                end_idx = min(start_idx + decode_batch_size, len(visual_imgs))
                batch_imgs = visual_imgs[start_idx:end_idx]
                batch_rewards = reward_model(batch_imgs, prompt)
                if not isinstance(batch_rewards, list):
                    batch_rewards = [batch_rewards]
                all_rewards.extend(batch_rewards)
            # rewards = reward_model(visual_imgs, prompt)
            rewards = all_rewards
            max_idx = np.argmax(np.array(rewards))
            # min_idx = np.argmin(np.array(rewards))
            # max_confidence = np.array(confidence[max_idx].cpu().float().numpy())
            # min_confidence = np.array(confidence[min_idx].cpu().float().numpy())
            # max_kl_seq = np.array(kl_seq[max_idx].cpu().float().numpy())
            # min_kl_seq = np.array(kl_seq[min_idx].cpu().float().numpy())
            # max_inter_score = 0.4*max_confidence + 0.25*max_kl_seq
            # min_inter_score = 0.4*min_confidence + 0.25*min_kl_seq
            # os.makedirs(f'./confidence_save/IR_{parallel_size}_bon_max',exist_ok=True)
            # os.makedirs(f'./confidence_save/IR_{parallel_size}_bon_min',exist_ok=True)  
            # np.save(os.path.join(f'./confidence_save/IR_{parallel_size}_bon_max',f'{idx}_confidence.npy'), max_confidence)
            # np.save(os.path.join(f'./confidence_save/IR_{parallel_size}_bon_max',f'{idx}_kl_seq.npy'), max_kl_seq)
            # np.save(os.path.join(f'./confidence_save/IR_{parallel_size}_bon_max',f'{idx}_inter_score.npy'), max_inter_score)
            # np.save(os.path.join(f'./confidence_save/IR_{parallel_size}_bon_min',f'{idx}_confidence.npy'), min_confidence)
            # np.save(os.path.join(f'./confidence_save/IR_{parallel_size}_bon_min',f'{idx}_kl_seq.npy'), min_kl_seq)
            # np.save(os.path.join(f'./confidence_save/IR_{parallel_size}_bon_min',f'{idx}_inter_score.npy'), min_inter_score)
            visual_imgs[max_idx].save(os.path.join(save_dir,f'{idx}.png'))
            # geneval_rewards = reward_model_2([visual_imgs[max_idx]], prompt, metadata=[metadata_list[idx]])

            max_reward = np.max(np.array(rewards))
            score+=max_reward

            # geneval_reward += np.max(np.array(geneval_rewards))


            with open(os.path.join(save_dir, 'reward.txt'), 'a') as f:
                f.write(f'{score}\n')
            
            # with open(os.path.join(save_dir, 'geneval_reward.txt'), 'a') as f:
            #     f.write(f'{geneval_reward}\n')
            
            clear_gpu_cache()