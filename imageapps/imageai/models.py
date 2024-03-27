from django.db import models
import numpy as np
import torch
from PIL import Image
import io,base64

# Create your models here.

class Photo(models.Model):
    image=models.ImageField(upload_to="photos")

    IMAGE_SIZE=100#画像サイズ
    MODEL_PATH="./imageai/ml_models/model-1.h5"
    imagename=[自分で作ったデータラベル]
    image_len=len(imagename)

    def predict(self):
        model=None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            model=torch.jit.load(self.MODEL_PATH, map_location=device)

            img_data=self.image.read()
            img_bin=io.BytesIO(img_data)

            image=Image.open(img_bin)
            image=image.convert("RGB")
            image=image.resize((self.IMAGE_SIZE,self.IMAGE_SIZE))
            data=np.asarray(image)/255.0
            X=torch.from_numpy(data).float().unsqueeze(0).to(device)

            result=model(X).cpu().numpy()[0]
            predicted=result.argmax()
            percentage=int(result[predicted]*100)

            return self.imagename[predicted],percentage
    def image_src(self):
        with self.image.open() as img:
            base64_img=base64.b64encode(img.read()).decode()

            return "data:"+img.file.content_type+";base64,"+base64_img
