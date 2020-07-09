验证码识别

依赖：

- captcha
- tensorflow 2.x
- numpy
- tqdm
- matplotlib
- pandas
- pydot-ng
- graphviz


## 运行方式

### 1、生成数据集

生成训练集

```shell script
python captcha_gen.py -c 10000 -m train
```

生成验证集

```shell script
python captcha_gen.py -c 1000 -m val
```

生成测试集

```shell script
python captcha_gen.py -c 1000 -m test
```



参考链接：

[ypwhs/captcha_break](https://github.com/ypwhs/captcha_break)