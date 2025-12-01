import torch.nn as nn
from basicsr.utils.registry import ARCH_REGISTRY
from PIL import Image
from mplug_owl2.model.builder import load_pretrained_model
import torch
from mplug_owl2.conversation import conv_templates
from mplug_owl2.mm_utils import tokenizer_image_token
from typing import List
import numpy as np

@ARCH_REGISTRY.register()
class ROC4MLLMArch(nn.Module):
    """架构说明文档"""

    def __init__(self, pretrained="", device="cuda:0",model=None,tokenizer=None,image_processor=None):
        super(ROC4MLLMArch, self).__init__()
        # 初始化网络层
        if model is None:
            tokenizer, model, image_processor, _ = load_pretrained_model(pretrained, None, "mplug_owl2", device=device)
        query = "<|image|>\nPlease rate the aesthetics of the image."
        conv = conv_templates["v1"].copy()
        roles = conv.roles
        inp = query
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()+'The aesthetic rate of the image is'
        self.input_ids = tokenizer_image_token(prompt, tokenizer, -200, return_tensors='pt').unsqueeze(0).to(
            model.device)
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor

    def expand2square(self, pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result
    def forward(self, image):
        #输入为图像list，图像为pil类型
        #输出为分数和文本，均为list类型
        image = [self.expand2square(img, tuple(int(x * 255) for x in self.image_processor.image_mean)) for img in image]
        with torch.inference_mode():
            image_tensors = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().to(
                self.model.device)
            # print(image_tensors.shape)
            # print(torch.cat(image_tensors, 0).shape)
            outputs = self.model.generate(
                self.input_ids.repeat(len(image_tensors), 1),
                images=image_tensors,
                do_sample=False,
                max_new_tokens=512,
                use_cache=True,
                output_hidden_states=True,
                return_dict_in_generate=True,
                output_scores=True,
            )
            output_text = []
            output_score = []
            logits = outputs.scores
            # print(len(image_tensors))
            for i in range(len(image_tensors)):
                output_ids = outputs.sequences[i]
                # special_token_index = (
                #         (output_ids >= self.model.config.output_first_id) & (
                #         output_ids <= self.model.config.output_last_id)).nonzero()
                special_token_index = (output_ids == self.model.config.score_id).nonzero()
                if len(special_token_index):
                    index = special_token_index[0, 0]
                    input_embedding = logits[index - self.input_ids.shape[1]][i,
                                      -self.model.config.img_token_num:].view(1, -1)
                    score = torch.softmax(input_embedding, dim=1)
                    w = torch.from_numpy(
                        np.linspace(self.model.config.min_score, self.model.config.max_score,
                                    self.model.config.num_tokens)).to(
                        score.device)
                    w = w.type(torch.FloatTensor)
                    w_batch = w.repeat(score.size(0), 1).to(score.device)

                    score = (score * w_batch).sum(dim=1)
                    text1 = self.tokenizer.decode(output_ids[self.input_ids.shape[1]:index], skip_special_tokens=True)
                    text2 = self.tokenizer.decode(output_ids[index + 1:], skip_special_tokens=True)
                    pred_text = text1 + f" {round(float(score), 4)} " + text2
                    output_text.append(pred_text)
                    output_score.append(round(float(score), 4))

                else:
                    pred_text = self.tokenizer.decode(output_ids[self.input_ids.shape[1]:],
                                                      skip_special_tokens=True).strip()
                    output_text.append(pred_text)
                    output_score.append(-1)
        return output_score,output_text