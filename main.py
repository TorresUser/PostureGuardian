# import os
# from src import pancho

# categories = ["buena_pose", "mala_pose"]

# for category in categories:
#     videos = os.listdir(f"./inputs/{category}")

#     for video in videos:
#         pancho.make_frames(f"./inputs/{category}/{video}", f"./inputs/{category}/frames", f"./trainer/dataset/{category}")
    

from trainer import trainer