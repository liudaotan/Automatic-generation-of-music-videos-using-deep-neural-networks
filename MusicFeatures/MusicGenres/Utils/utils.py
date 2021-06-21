import os
folders = list(filter(lambda x: os.path.isdir(os.path.join("fma_small/",x)), os.listdir("fma_small/")))
