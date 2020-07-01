import string

WIDTH =  170 # 验证码宽
HEIGHT = 80 # 验证码高

CAPTCHA_LEN =  4 # 验证码个数
CAPTCHAES = string.digits+string.ascii_uppercase # 字符种类，默认数字+大写字母
CAPTCHA_CLASSES = len(CAPTCHAES) # 字符类别 # 默认10+26

TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
TEST_DIR = "data/test"