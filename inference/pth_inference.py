import torch.onnx
import torch.nn as nn
from utils import image_normalize, input_dim_3to4
from config import imagenet_classes, input_size_dict
import numpy as np
import os
import torchvision.models as model
import sys
import cv2

model_folder = "../models"


class pthInference:
    def __init__(self, model_name, plot_res=True):
        self.device = "cuda:0"
        self.model_name = model_name + ".pth"
        self.model = self.__get_model(model_name)
        self.model.load_state_dict(torch.load(os.path.join(model_folder, "pth/{}".format(self.model_name))))
        self.model.cuda()
        self.size = input_size_dict[model_name]
        self.plot = plot_res

    @staticmethod
    def __get_model(name):
        if "resnet18" in name:
            return model.resnet18()
        elif "resnet50" in name:
            return model.resnet50()
        elif "resnet34" in name:
            return model.resnet34()
        elif "resnet101" in name:
            return model.resnet101()
        elif "resnet152" in name:
            return model.resnet152()
        elif "inception" in name:
            return model.inception_v3()
        elif "mobilenet" in name:
            return model.mobilenet_v2()
        elif "shufflenet" in name:
            return model.shufflenet_v2_x1_0()
        elif "squeezenet" in name:
            return model.squeezenet1_1()
        else:
            raise ValueError("Wrong name of pre-train model, please check the model name")

    def __test_model(self, input_tensor):
        self.model.eval()
        image_batch_tensor = input_tensor.cuda()
        outputs = self.model(image_batch_tensor)
        outputs_tensor = outputs.data
        m_softmax = nn.Softmax(dim=1)
        outputs_tensor = m_softmax(outputs_tensor).to("cpu")
        return np.asarray(outputs_tensor)

    def predict(self, img_path):
        input_tensor = input_dim_3to4(image_normalize(img_path, size=self.size))
        output = self.__test_model(input_tensor)
        # print(output)
        idx = output[0].tolist().index(max(output[0].tolist()))
        print("Predicted image is {}".format(img_path.split("\\")[-1]))
        print("Predicted index is {}".format(idx))
        print("Predicted classes is {}".format(imagenet_classes[idx]))
        print("The score is {}\n\n".format(output[0][idx]))

        if self.plot:
            img = cv2.imread(img_path)
            cv2.putText(img, self.model_name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, imagenet_classes[idx], (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img, str(output[0][idx]), (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("img", img)
            cv2.waitKey(1500)


if __name__ == '__main__':
    img_folder = "../image"
    nets = ["mobilenet", "resnet18", "squeezenet"]
    image_ls = [os.path.join(img_folder, image_path) for image_path in os.listdir(img_folder)]
    for mdl in nets:
        pth = pthInference(mdl)
        for image in image_ls:
            pth.predict(image)
