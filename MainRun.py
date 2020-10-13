from librosa_play import *
#import Record_Play

# Record_Play.record_play('lulala\lulala_yuan.wav','lulala\lulala_record.wav')
# Record_Play.play('lulala\lulala_record.wav')

music_compare=wave_compare_show(filename_record=r'lulala\lulala_chen.wav',filename_music=r'lulala\lulala_origin.wav')
# 偏移修正，范围默认100
offset=music_compare.OffSet()
print('偏移秒数='+str(offset/44100))#大致偏移秒数，采样率44100
#音高展示
Pitch_Similarity=music_compare.Pitch_Step_Show()
print("Pitch_Similarity="+str(Pitch_Similarity))
#节奏展示
tempo_music,tempo_record = music_compare.Tempogram_Show1()
print('tempo_music='+str(tempo_music)+',tempo_record='+str(tempo_record))
music_compare.Tempogram_Show2()

# 分贝展示
DB_Similarity= music_compare.DB_Power_Show()
print("DB_Similarity="+str(DB_Similarity))
music_compare.DB_Log_Power_Show()








