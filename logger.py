class Logger:
    def __init__(self, path, resume):
        self.path = path + "/" + "log.txt"
        if not resume:
            f = open(self.path, "w")
            f.close()

    def log(self, epoch, train_info, val_info):
        message = "{0}:train {1}, val :{2}\n".format(epoch, train_info, val_info)
        with open(self.path, "a") as  file:
            file.write(message)