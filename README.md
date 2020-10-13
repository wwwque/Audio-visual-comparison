# Audio visual comparison 
Innovation project for College Students


# 可视化音频对比
大学生创新项目：KTV打分算法

整合了librosa库的部分功能来提取音频特征进行音频可视化对比操作

--------------------
- import librosa
- import matplotlib.pyplot as plt
- import librosa.display
- import numpy as np
- import scipy.stats

## class类  ##
### librosa_play.wave_compare_show(filename_record,filename_music)
只需简单输入两个音频路径实例化即可（测试用例均采用采样率44100的音频，其它采样率未测试）

### wave\_compare_show的核心函数（需先实例化）
--------------------

#### OffSet(offset\_range=100): 音频同步
return 偏移量(int)（单位：秒数*采样率）

- 说明：以音高为基准对两个音频进行同步操作，
- offset_range：偏移的范围
- 建议使用其它函数前都执行一下此操作

#### Pitch\_Step_Show(self,show=True): 音高对比
return 相似度(double)

- 说明：展示两个音频的最明显的音高的特征，返回大致相似度
- show：显示图，设置成False则不显示

#### DB\_Power_Show(show=True): 分贝对比
return 相似度(double)

- 说明：展示两个音频的分贝特征，返回大致相似度
- show：显示图，设置成False则不显示

#### Tempogram\_Show1(self,show=True): 节拍对比
return tempo_music(double),tempo_record(double)

- 说明：展示两个音频的节奏特征，返回两个音频的节拍
- show：显示图，设置成False则不显示

其它函数可不用理会


## class类  ##
### wave\_show(filename) 
对单个音频进行特征图展示，暂不说明(
