from PIL import Image

import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from BLIP.models.blip import blip_decoder


class Image2Text:

    def __init__(self, config, model, device):
        # Load the model
        model = blip_decoder(pretrained=config.model_url,
                             image_size=config.image_size,
                             vit='base')
        model.eval()
        self.model = model.to(device)
        self.config = config
        self.device = device

    def generate_caption(self, raw_image):
        image = self.preprocess(raw_image)
        with torch.no_grad():
            # beam search
            caption = self.model.generate(image,
                                          sample=False,
                                          num_beams=3,
                                          max_length=20,
                                          min_length=5)
            # nucleus sampling
            # caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5)
            print('caption: ' + caption[0])

    def preprocess(self, raw_image):
        w, h = raw_image.size
        #display(raw_image.resize((w//5,h//5)))

        transform = transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size),
                              interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711))
        ])
        image = transform(raw_image).unsqueeze(0).to(self.device)
        return image
