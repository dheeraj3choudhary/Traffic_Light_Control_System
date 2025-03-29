import configparser

cfg = configparser.ConfigParser()
cfg.read(r"C:\Users\jaivb\Desktop\Traffic_Light_Control_System\configs\training_settings.ini")
print(cfg.sections())
