import torch.onnx
import torch.nn as nn
from utils import image_normalize, input_dim_3to4
from config import imagenet_classes
import numpy as np
import os
from torchvision.models import squeezenet1_1 as squeezenet
import cv2

model_folder = "../../models"


class pthTester:
    def __init__(self, img_path, plot_res=True, to_onnx=False):
        self.device = "cuda:0"
        self.model = squeezenet()
        self.model_name = "squeezenet.pth"
        self.model.load_state_dict(torch.load(os.path.join(model_folder, "pth/{}".format(self.model_name))))
        self.model.cuda()
        self.to_onnx = to_onnx
        self.input_tensor = input_dim_3to4(image_normalize(img_path))
        self.plot = plot_res
        self.img = cv2.imread(img_path)

    def __test_model(self):
        self.model.eval()
        image_batch_tensor = self.input_tensor.cuda()
        outputs = self.model(image_batch_tensor)
        outputs_tensor = outputs.data
        m_softmax = nn.Softmax(dim=1)
        outputs_tensor = m_softmax(outputs_tensor).to("cpu")
        return np.asarray(outputs_tensor)

    def predict(self):
        output = self.__test_model()
        # print(output)
        idx = output[0].tolist().index(max(output[0].tolist()))
        print("Predicted index is {}".format(idx))
        print("Predicted classes is {}".format(imagenet_classes[idx]))
        print("The possibility is {}".format(output[0][idx]))

        if self.to_onnx:
            dest_folder = os.path.join(model_folder, "onnx")
            os.makedirs(dest_folder, exist_ok=True)
            torch_out = torch.onnx.export(self.model, self.input_tensor.cuda(),
                                          os.path.join(dest_folder, self.model_name.replace("pth", "onnx")), verbose=False)

        if self.plot:
            cv2.putText(self.img, imagenet_classes[idx], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(self.img, str(output[0][idx]), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("img", self.img)
            cv2.waitKey(0)


if __name__ == '__main__':
    img_path = "../../image/cat1.jpg"
    pthTester(img_path, True).predict()
