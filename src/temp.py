import configparser

cfg = configparser.ConfigParser()
cfg.read(r"configs\training_settings.ini")
print(cfg.sections())

# configs\training_settings.ini