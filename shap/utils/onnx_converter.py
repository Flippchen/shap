import onnx2tf
from onnx2pytorch import ConvertModel

from shap.explainers._deep import PyTorchDeep, TFDeep


def convert_onnx_model(model, data, session, learning_phase_flags):

    try:
        model = ConvertModel(model)
        return PyTorchDeep(model, data)
    except:
        print("Error converting onnx model to pytorch")

   #try:
   #    model = onnx2tf.convert(onnx_graph=model)
   #    return TFDeep(model, data, session, learning_phase_flags)
   #except:
   #    print("Error converting onnx model to tensorflow")

   #raise Exception("Error converting onnx model to tensorflow")