import string
import os

WIDTH =  128 # 验证码宽
HEIGHT = 64 # 验证码高

CAPTCHA_LEN =  4 # 验证码个数
CAPTCHAES = string.digits+string.ascii_uppercase # 字符种类，默认数字+大写字母
CAPTCHA_CLASSES = len(CAPTCHAES) # 字符类别 # 默认10+26

TRAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),"data/train"))
VAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),"data/val"))
TEST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),"data/test"))

SAVE_FREQ = 2 # 保存模型的频率

