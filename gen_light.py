import onnx
from onnx import helper, TensorProto

# Définition de l'entrée et de la sortie (1, 3, 224, 224)
input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 224, 224])
output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 224, 224])

# Opération simple (Identity)
node_def = helper.make_node('Identity', ['input'], ['output'])

# Assemblage du graphe
graph_def = helper.make_graph([node_def], 'test-pipeline', [input_tensor], [output_tensor])
model_def = helper.make_model(graph_def, producer_name='onnx-light')

# FORCER LA VERSION IR COMPATIBLE AVEC ONNX RUNTIME 1.17.1 (IR v9 / Opset 19)
model_def.ir_version = 9
model_def.opset_import[0].version = 19

# Sauvegarde
onnx.save(model_def, 'model.onnx')
print("Fichier model.onnx (IR v9) généré avec succès !")
