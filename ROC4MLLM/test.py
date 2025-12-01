from mplug_owl2.assessor import Assessment
from PIL import Image

assessment=Assessment(pretrained="../ROC4MLLM_weights")
images=["test_images/1_-10.jpg","test_images/1_-10.jpg"]
input_img=[]
for image in images:
    img=Image.open(image).convert('RGB')
    input_img.append(img)
answer=assessment(input_img,precision=4)
print(answer)




