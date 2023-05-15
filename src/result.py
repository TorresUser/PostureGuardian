# Procesa la precisión promedio del todo el test y guarda el result.json (como informe del test)y actualiza data.json (como informe de todos los tests).
# result.json:
# {
#   "name": "test0001",
#   "date": "2021-08-31 18:00:00",
#   "accuracy": 98,
#   "execution_time": 0.5,
#   "total_images": 20,
#   "cat1_images": 10,
#   "cat2_images": 10,
#   "pancho": :hotdog:
# }
# data.json:
# {
#   "average_accuracy": 98,
#   "average_execution_time": 0.5,
#     "test0001":{
#         "Accuracy": 98,
#         "total_images": 100,
#         "cat1_images": 50,
#         "cat2_images": 50,
#         "execution_time": 0.5,
#         "date": "2019-01-01 18:00:00"
#     },
#     "test0002":{
#         "Accuracy": 97,
#         "total_images": 100,
#         "cat1_images": 50,
#         "cat2_images": 50,
#         "execution_time": 0.5,
#         "date": "2019-01-01 18:00:00"
#     },
#     "test0003":{
#         "Accuracy": 99,
#         "total_images": 100,
#         "cat1_images": 50,
#         "cat2_images": 50,
#         "execution_time": 0.6,
#         "date": "2019-01-01 18:00:00"
#     }
# }