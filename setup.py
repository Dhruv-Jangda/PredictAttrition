from os import mkdir

paths = {
    "processed": ".\\data\\processed",
    "result": ".\\data\\result",
    "model": ".\\data\\model",
    "logs": ".\\data\\logs",
    "app_logs": ".\\data\\logs\\app",
    "mxboard_logs": ".\\data\\logs\\tensorboard"
}

for key in paths.keys():
    mkdir(path=paths[key])
