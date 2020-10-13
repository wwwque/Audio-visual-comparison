import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import scipy.stats



class wave_show():
#单个音频特征图显示
    def __init__(self,filename):
        self.filename=filename
        self.__y, self.__sr = librosa.load(filename,sr=None)
        # extract mel spectrogram feature
        self.__melspec = librosa.feature.melspectrogram(self.__y, self.__sr, n_fft=1024, hop_length=512, n_mels=128)
        # convert to log scale
        self.__logmelspec = librosa.power_to_db(self.__melspec)
        #跟踪节奏
        self.__onset_env = librosa.onset.onset_strength(self.__y, sr=self.__sr)
        self.__tempo, self.__beats = librosa.beat.beat_track(onset_envelope=self.__onset_env,
                                                sr=self.__sr)
        self.__pulse = librosa.beat.plp(onset_envelope=self.__onset_env, sr=self.__sr)
        # # Or compute pulse with an alternate prior, like log-normal
        #
        self.__prior = scipy.stats.lognorm(loc=np.log(120), scale=120, s=1)
        self.__pulse_lognorm = librosa.beat.plp(onset_envelope=self.__onset_env, sr=self.__sr,
                                          prior=self.__prior)
        self.__hop_length = 512


    #分贝展示
    def DB_Show(self):
        S = np.abs(librosa.stft(self.__y))

        fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
        imgpow = librosa.display.specshow(S ** 2, sr=self.__sr, y_axis='log', x_axis='time',
                                          ax=ax[0])
        ax[0].set(title='Power spectrogram')
        ax[0].label_outer()
        imgdb = librosa.display.specshow(librosa.power_to_db(S ** 2, ref=np.max),
                                         sr=self.__sr, y_axis='log', x_axis='time', ax=ax[1])
        ax[1].set(title='Log-Power spectrogram')
        fig.colorbar(imgpow, ax=ax[0])
        fig.colorbar(imgdb, ax=ax[1], format="%+2.0f dB")
        plt.show()

    #节奏展示
    def Tempogram_Show(self):
        tempogram = librosa.feature.tempogram(onset_envelope=self.__onset_env, sr=self.__sr,
                                              hop_length=self.__hop_length)
        ac_global = librosa.autocorrelate(self.__onset_env, max_size=tempogram.shape[0])
        ac_global = librosa.util.normalize(ac_global)

        tempo = librosa.beat.tempo(onset_envelope=self.__onset_env, sr=self.__sr,
                                   hop_length=self.__hop_length)[0]

        fig, ax = plt.subplots(nrows=4, figsize=(10, 10))
        times = librosa.times_like(self.__onset_env, sr=self.__sr, hop_length=self.__hop_length)
        ax[0].plot(times, self.__onset_env, label='Onset strength')
        ax[0].label_outer()
        ax[0].legend(frameon=True)
        librosa.display.specshow(tempogram, sr=self.__sr, hop_length=self.__hop_length,
                                 x_axis='time', y_axis='tempo', cmap='magma',
                                 ax=ax[1])
        ax[1].axhline(tempo, color='w', linestyle='--', alpha=1,
                      label='Estimated tempo={:g}'.format(tempo))
        ax[1].legend(loc='upper right')
        ax[1].set(title='Tempogram')
        x = np.linspace(0, tempogram.shape[0] * float(self.__hop_length) / self.__sr,
                        num=tempogram.shape[0])
        ax[2].plot(x, np.mean(tempogram, axis=1), label='Mean local autocorrelation')
        ax[2].plot(x, ac_global, '--', alpha=0.75, label='Global autocorrelation')
        ax[2].set(xlabel='Lag (seconds)')
        ax[2].legend(frameon=True)
        freqs = librosa.tempo_frequencies(tempogram.shape[0], hop_length=self.__hop_length, sr=self.__sr)
        ax[3].semilogx(freqs[1:], np.mean(tempogram[1:], axis=1),
                       label='Mean local autocorrelation', basex=2)
        ax[3].semilogx(freqs[1:], ac_global[1:], '--', alpha=0.75,
                       label='Global autocorrelation', basex=2)
        ax[3].axvline(tempo, color='black', linestyle='--', alpha=.8,
                      label='Estimated tempo={:g}'.format(tempo))
        ax[3].legend(frameon=True)
        ax[3].set(xlabel='BPM')
        ax[3].grid(True)
        plt.show()

    #音高展示
    def Pitches_Show(self):
        # 音色谱
        chroma_stft = librosa.feature.chroma_stft(y=self.__y, sr=self.__sr, n_chroma=12, n_fft=4096)
        # 另一种常数Q音色谱
        chroma_cq = librosa.feature.chroma_cqt(y=self.__y, sr=self.__sr)
        # 功率归一化音色谱
        chroma_cens = librosa.feature.chroma_cens(y=self.__y, sr=self.__sr)

        plt.figure(figsize=(15, 15))
        plt.subplot(3, 1, 1)
        librosa.display.specshow(chroma_stft, y_axis='chroma')
        plt.title('chroma_stft')
        plt.colorbar()
        plt.subplot(3, 1, 2)
        librosa.display.specshow(chroma_cq, y_axis='chroma', x_axis='time')
        plt.title('chroma_cqt')
        plt.colorbar()
        plt.subplot(3, 1, 3)
        librosa.display.specshow(chroma_cens, y_axis='chroma', x_axis='time')
        plt.title('chroma_cens')
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    #节拍展示
    def Beat_PLP_Show(self):#主要本地脉冲PLP估计
        fig, ax = plt.subplots(nrows=3, sharex=True)
        librosa.display.specshow(librosa.power_to_db(self.__melspec,
                                                     ref=np.max),
                                 x_axis='time', y_axis='mel', ax=ax[0])
        ax[0].set(title='Mel spectrogram')
        ax[0].label_outer()
        ax[1].plot(librosa.times_like(self.__onset_env),
                 librosa.util.normalize(self.__onset_env),
                 label='Onset strength')
        ax[1].plot(librosa.times_like(self.__pulse),
                 librosa.util.normalize(self.__pulse),
                 label='Predominant local pulse (PLP)')
        ax[1].set(title='Uniform tempo prior [30, 300]')
        ax[1].label_outer()
        ax[2].plot(librosa.times_like(self.__onset_env),
                 librosa.util.normalize(self.__onset_env),
                 label='Onset strength')
        ax[2].plot(librosa.times_like(self.__pulse_lognorm),
                 librosa.util.normalize(self.__pulse_lognorm),
                 label='Predominant local pulse (PLP)')
        ax[2].set(title='Log-normal tempo prior, mean=120', xlim=[5, 20])
        ax[2].legend()
        plt.tight_layout()
        plt.show()

    def Beat_Track_Show(self):#节拍跟踪器
        hop_length = 512
        fig, ax = plt.subplots(nrows=2, sharex=True)
        times = librosa.times_like(self.__onset_env, sr=self.__sr, hop_length=hop_length)
        M = librosa.feature.melspectrogram(y=self.__y, sr=self.__sr, hop_length=hop_length)
        # librosa.display.specshow(librosa.power_to_db(M, ref=np.max),
        #                          y_axis='mel', x_axis='time', hop_length=hop_length,
        #                          ax=ax[0])
        librosa.display.specshow(self.__logmelspec, sr=self.__sr, x_axis='time', y_axis='mel',ax=ax[0])
        ax[0].label_outer()
        ax[0].set(title='Mel spectrogram')
        ax[1].plot(times, librosa.util.normalize(self.__onset_env),
                   label='Onset strength')
        ax[1].vlines(times[self.__beats], 0, 1, alpha=0.5, color='r',
                     linestyle='--', label='Beats')
        ax[1].legend()
        plt.show()

    def Beat_Wavform(self):
        plt.figure()
        plt.subplot(2, 1, 1)
        librosa.display.waveplot(self.__y, self.__sr)
        plt.title('Beat wavform')

        plt.subplot(2, 1, 2)
        librosa.display.specshow(self.__logmelspec, sr=self.__sr, x_axis='time', y_axis='mel')
        plt.title('Mel spectrogram')
        plt.tight_layout()
        plt.show()



class wave_compare_show():
#两个音频特征图对比
    def __init__(self,filename_record,filename_music):
        self.__filename_record=filename_record
        self.__y_record, self.__sr_record = librosa.load(filename_record,sr=None)
        # extract mel spectrogram feature
        self.__melspec_record = librosa.feature.melspectrogram(self.__y_record, self.__sr_record, n_fft=1024, hop_length=512, n_mels=128)
        # convert to log scale
        self.__logmelspec_record = librosa.power_to_db(self.__melspec_record)
        #跟踪节奏
        self.__onset_env_record = librosa.onset.onset_strength(self.__y_record, sr=self.__sr_record)
        self.__tempo_record, self.__beats_record = librosa.beat.beat_track(onset_envelope=self.__onset_env_record,
                                                sr=self.__sr_record)
        self.__pulse_record = librosa.beat.plp(onset_envelope=self.__onset_env_record, sr=self.__sr_record)
        # # Or compute pulse with an alternate prior, like log-normal
        #
        self.__prior_record = scipy.stats.lognorm(loc=np.log(120), scale=120, s=1)
        self.__pulse_lognorm_record = librosa.beat.plp(onset_envelope=self.__onset_env_record, sr=self.__sr_record,
                                          prior=self.__prior_record)
        ############################
        self.__filename_music = filename_music
        self.__y_music, self.__sr_music = librosa.load(filename_music, sr=None)
        # extract mel spectrogram feature
        self.__melspec_music = librosa.feature.melspectrogram(self.__y_music, self.__sr_music, n_fft=1024, hop_length=512, n_mels=128)
        # convert to log scale
        self.__logmelspec_music = librosa.power_to_db(self.__melspec_music)
        # 跟踪节奏
        self.__onset_env_music = librosa.onset.onset_strength(self.__y_music, sr=self.__sr_music)
        self.__tempo_music, self.__beats_music = librosa.beat.beat_track(onset_envelope=self.__onset_env_music,
                                                         sr=self.__sr_music)
        self.__pulse_music = librosa.beat.plp(onset_envelope=self.__onset_env_music, sr=self.__sr_music)
        # # Or compute pulse with an alternate prior, like log-normal
        #
        self.__prior_music = scipy.stats.lognorm(loc=np.log(120), scale=120, s=1)
        self.__pulse_lognorm_music = librosa.beat.plp(onset_envelope=self.__onset_env_music, sr=self.__sr_music,
                                              prior=self.__prior_music)

    def OffSet1(self):#暴力偏移：耗时
        y_record_origin=self.__y_record
        y_record_best=self.__y_record
        similarity_best=0
        for ofs in range(0,101):
            self.__y_record=self.__y_record[512:]
            similarity=wave_compare_show.Pitch_Step_Show(self,False)
            if similarity_best<similarity:
                similarity_best=similarity
                y_record_best=self.__y_record
        # self.__y_record=y_record_origin
        # for ofs in range(0,101):
        #     self.__y_record=self.__y_record[ofs:]
        #     similarity=wave_compare_show.Pitch_Step_Show(self,False)
        #     if similarity_best<similarity:
        #         similarity_best=similarity
        #         y_record_best=self.__y_record
        self.__y_record=y_record_best
        # <editor-fold desc="重置">
        self.__melspec_record = librosa.feature.melspectrogram(self.__y_record, self.__sr_record, n_fft=1024, hop_length=512, n_mels=128)
        # convert to log scale
        self.__logmelspec_record = librosa.power_to_db(self.__melspec_record)
        #跟踪节奏
        self.__onset_env_record = librosa.onset.onset_strength(self.__y_record, sr=self.__sr_record)
        self.__tempo_record, self.__beats_record = librosa.beat.beat_track(onset_envelope=self.__onset_env_record,
                                                sr=self.__sr_record)
        self.__pulse_record = librosa.beat.plp(onset_envelope=self.__onset_env_record, sr=self.__sr_record)
        # # Or compute pulse with an alternate prior, like log-normal
        #
        self.__prior_record = scipy.stats.lognorm(loc=np.log(120), scale=120, s=1)
        self.__pulse_lognorm_record = librosa.beat.plp(onset_envelope=self.__onset_env_record, sr=self.__sr_record,
                                          prior=self.__prior_record)

        self.__melspec_music = librosa.feature.melspectrogram(self.__y_music, self.__sr_music, n_fft=1024, hop_length=512, n_mels=128)
        # convert to log scale
        self.__logmelspec_music = librosa.power_to_db(self.__melspec_music)
        # 跟踪节奏
        self.__onset_env_music = librosa.onset.onset_strength(self.__y_music, sr=self.__sr_music)
        self.__tempo_music, self.__beats_music = librosa.beat.beat_track(onset_envelope=self.__onset_env_music,
                                                         sr=self.__sr_music)
        self.__pulse_music = librosa.beat.plp(onset_envelope=self.__onset_env_music, sr=self.__sr_music)
        # # Or compute pulse with an alternate prior, like log-normal
        #
        self.__prior_music = scipy.stats.lognorm(loc=np.log(120), scale=120, s=1)
        self.__pulse_lognorm_music = librosa.beat.plp(onset_envelope=self.__onset_env_music, sr=self.__sr_music,
                                              prior=self.__prior_music)
        # </editor-fold>

    def OffSet(self,offset_range=100):#根据节奏设置偏移
        chroma_cq_record = librosa.feature.chroma_cqt(y=self.__y_record, sr=self.__sr_record)
        chroma_cq_music = librosa.feature.chroma_cqt(y=self.__y_music, sr=self.__sr_music)
        music_length = min(len(chroma_cq_record[0]), len(chroma_cq_music[0]),)
        music_length = min(music_length,500)

        pitch_record = np.zeros(music_length)
        pitch_music = np.zeros(music_length)
        Difference = 0
        offset=-10

        for i in range(0, music_length):
            for j in range(0, 12):
                if chroma_cq_record[j][i] == 1:
                    pitch_record[i] = j
                if chroma_cq_music[j][i] == 1:
                    pitch_music[i] = j


        maximun=0
        for offs in range(0,offset_range+1):
            lright = 0
            rright = 0
            for i in range(0,music_length-offset_range-2):
                if pitch_music[i]==pitch_record[i+offs]:
                    lright+=1
                if pitch_music[i+offs]==pitch_record[i]:
                    rright+=1
            if maximun<lright:
                maximun=lright
                offset=offs
            if maximun<rright:
                maximun=rright
                offset=-offs

        x=0
        t=1
        if offset>=0:
            x = len(self.__y_record)/len(self.__onset_env_record)*offset
            x=int(x)
            self.__y_record = self.__y_record[x:]
        elif offset<0:
            t=-1
            x=len(self.__y_music)/len(self.__onset_env_music)*-offset
            x = int(x)
            self.__y_music = self.__y_music[x:]

        # <editor-fold desc="重置">
        self.__melspec_record = librosa.feature.melspectrogram(self.__y_record, self.__sr_record, n_fft=1024,
                                                               hop_length=512, n_mels=128)
        # convert to log scale
        self.__logmelspec_record = librosa.power_to_db(self.__melspec_record)
        # 跟踪节奏
        self.__onset_env_record = librosa.onset.onset_strength(self.__y_record, sr=self.__sr_record)
        self.__tempo_record, self.__beats_record = librosa.beat.beat_track(onset_envelope=self.__onset_env_record,
                                                                           sr=self.__sr_record)
        self.__pulse_record = librosa.beat.plp(onset_envelope=self.__onset_env_record, sr=self.__sr_record)
        # # Or compute pulse with an alternate prior, like log-normal
        #
        self.__prior_record = scipy.stats.lognorm(loc=np.log(120), scale=120, s=1)
        self.__pulse_lognorm_record = librosa.beat.plp(onset_envelope=self.__onset_env_record, sr=self.__sr_record,
                                                       prior=self.__prior_record)

        self.__melspec_music = librosa.feature.melspectrogram(self.__y_music, self.__sr_music, n_fft=1024,
                                                              hop_length=512, n_mels=128)
        # convert to log scale
        self.__logmelspec_music = librosa.power_to_db(self.__melspec_music)
        # 跟踪节奏
        self.__onset_env_music = librosa.onset.onset_strength(self.__y_music, sr=self.__sr_music)
        self.__tempo_music, self.__beats_music = librosa.beat.beat_track(onset_envelope=self.__onset_env_music,
                                                                         sr=self.__sr_music)
        self.__pulse_music = librosa.beat.plp(onset_envelope=self.__onset_env_music, sr=self.__sr_music)
        # # Or compute pulse with an alternate prior, like log-normal
        #
        self.__prior_music = scipy.stats.lognorm(loc=np.log(120), scale=120, s=1)
        self.__pulse_lognorm_music = librosa.beat.plp(onset_envelope=self.__onset_env_music, sr=self.__sr_music,
                                                      prior=self.__prior_music)
        # </editor-fold>

        return x*t


    # <editor-fold desc="分贝特征展示">
    def DB_Power_Show(self,show=True):
        S_record = np.abs(librosa.stft(self.__y_record))
        S_music = np.abs(librosa.stft(self.__y_music))



        s_record=S_record**2
        s_music=S_music**2
        for i in range(len(s_record)):
            for j in range(len(s_record[0])):
                if s_record[i][j]<1000:
                    s_record[i][j]=0
        for i in range(len(s_music)):
            for j in range(len(s_music[0])):
                if s_music[i][j]<1000:
                    s_music[i][j]=0

        xlen=min(len(s_music),len(s_record))
        ylen=min(len(s_music[0]),len(s_record[0]))
        rigth=0
        sum=0
        for i in range(xlen):
            for j in range(ylen):
                if s_music[i][j]!=0:
                    sum+=1
                    for k in range(-2,2):
                        b=0
                        for l in range(-5, 5):
                            if k+i>=0 and k+i<xlen and l+j>=0 and l+j<ylen:
                                if s_record[k+i][l+j]!=0:
                                    rigth+=1
                                    b=1
                                    break
                        if b==1:
                            break


        Similarity=round(rigth/sum,3)
        if show==True:
            fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
            imgpow_record = librosa.display.specshow(s_record ** 2, sr=self.__sr_record, y_axis='log', x_axis='time',
                                              ax=ax[0])
            ax[0].set(title='Recording Power spectrogram')
            ax[0].label_outer()
            fig.colorbar(imgpow_record, ax=ax[0])


            imgpow_music = librosa.display.specshow(s_music ** 2, sr=self.__sr_music, y_axis='log', x_axis='time',
                                              ax=ax[1])
            ax[1].set(title='Music Power spectrogram')
            ax[1].label_outer()
            fig.colorbar(imgpow_music, ax=ax[1])
            plt.tight_layout()

            plt.show()
        return Similarity

    def DB_Log_Power_Show(self):
        S_record = np.abs(librosa.stft(self.__y_record))
        S_music = np.abs(librosa.stft(self.__y_music))


        fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
        imgdb_record = librosa.display.specshow(librosa.power_to_db(S_record ** 2, ref=np.max),
                                         sr=self.__sr_record, y_axis='log', x_axis='time', ax=ax[0])
        ax[0].set(title='Recording Log-Power spectrogram')
        fig.colorbar(imgdb_record, ax=ax[0], format="%+2.0f dB")

        imgdb_music = librosa.display.specshow(librosa.power_to_db(S_music ** 2, ref=np.max),
                                         sr=self.__sr_music, y_axis='log', x_axis='time', ax=ax[1])
        ax[1].set(title='Music Log-Power spectrogram')
        fig.colorbar(imgdb_music, ax=ax[1], format="%+2.0f dB")
        plt.tight_layout()
        plt.show()
    # </editor-fold>

    # <editor-fold desc="节奏特征图展示">
    def Tempogram_Show1(self,show=True):
        hop_length=512
        tempogram_record = librosa.feature.tempogram(onset_envelope=self.__onset_env_record, sr=self.__sr_record,
                                              hop_length=hop_length)
        tempogram_music = librosa.feature.tempogram(onset_envelope=self.__onset_env_music, sr=self.__sr_music,
                                    hop_length=hop_length)

        tempo_record = librosa.beat.tempo(onset_envelope=self.__onset_env_record, sr=self.__sr_record,
                                   hop_length=hop_length)[0]
        tempo_music = librosa.beat.tempo(onset_envelope=self.__onset_env_music, sr=self.__sr_music,
                                    hop_length=hop_length)[0]




        times_record = librosa.times_like(self.__onset_env_record, sr=self.__sr_record, hop_length=hop_length)
        times_music = librosa.times_like(self.__onset_env_music, sr=self.__sr_music, hop_length=hop_length)

        ###################
        if show==True:
            fig, ax = plt.subplots(nrows=3)


            ax[0].plot(times_record, self.__onset_env_record,':b', label='Recording Onset strength')
            ax[0].plot(times_music, self.__onset_env_music,':g', label='Music Onset strength')

            ax[0].label_outer()
            ax[0].legend(frameon=True)


            librosa.display.specshow(tempogram_record, sr=self.__sr_record, hop_length=hop_length,
                                     x_axis='time', y_axis='tempo', cmap='magma',
                                     ax=ax[1])
            ax[1].axhline(tempo_record, color='w', linestyle='--', alpha=1,
                          label='Estimated tempo={:g}'.format(tempo_record))
            ax[1].legend(loc='upper right')
            ax[1].set(title='Recording Tempogram')

    ###################

            librosa.display.specshow(tempogram_music, sr=self.__sr_music, hop_length=hop_length,
                                     x_axis='time', y_axis='tempo', cmap='magma',
                                     ax=ax[2])
            ax[2].axhline(tempo_music, color='w', linestyle='--', alpha=1,
                          label='Estimated tempo={:g}'.format(tempo_music))
            ax[2].legend(loc='upper right')
            ax[2].set(title='Music Tempogram')

            plt.tight_layout()
            plt.show()
        return format(tempo_music),format(tempo_record)

    def Tempogram_Show2(self,show=True):
        hop_length=512

        tempogram_record = librosa.feature.tempogram(onset_envelope=self.__onset_env_record, sr=self.__sr_record,
                                              hop_length=hop_length)

        tempo_record = librosa.beat.tempo(onset_envelope=self.__onset_env_record, sr=self.__sr_record,
                                   hop_length=hop_length)[0]

        ac_global_record = librosa.autocorrelate(self.__onset_env_record, max_size=tempogram_record.shape[0])
        ac_global_record = librosa.util.normalize(ac_global_record)

        x_record = np.linspace(0, tempogram_record.shape[0] * float(hop_length) / self.__sr_record,
                        num=tempogram_record.shape[0])



        tempogram_music = librosa.feature.tempogram(onset_envelope=self.__onset_env_music, sr=self.__sr_music,
                                    hop_length=hop_length)

        tempo_music = librosa.beat.tempo(onset_envelope=self.__onset_env_music, sr=self.__sr_music,
                                    hop_length=hop_length)[0]

        ac_global_music = librosa.autocorrelate(self.__onset_env_music, max_size=tempogram_music.shape[0])
        ac_global_music = librosa.util.normalize(ac_global_music)

        x_music = np.linspace(0, tempogram_music.shape[0] * float(hop_length) / self.__sr_music,
                        num=tempogram_music.shape[0])

        if show==True:
            fig,ax=plt.subplots(nrows=4)

            plt.title('Recording:')
            ax[0].plot(x_record, np.mean(tempogram_record, axis=1), label='Mean local autocorrelation')
            ax[0].plot(x_record, ac_global_record, '--', alpha=0.75, label='Global autocorrelation')

            ax[0].set(xlabel='Record Lag (seconds)')
            ax[0].legend(title='Recording:',frameon=True)

            ax[1].plot(x_music, np.mean(tempogram_music, axis=1), label='Mean local autocorrelation')
            ax[1].plot(x_music, ac_global_music, '--', alpha=0.75, label='Global autocorrelation')
            ax[1].set(xlabel='Music Lag (seconds)')
            ax[1].legend(title='Music:',frameon=True)


            freqs_record = librosa.tempo_frequencies(tempogram_record.shape[0], hop_length=hop_length, sr=self.__sr_record)
            freqs_music = librosa.tempo_frequencies(tempogram_music.shape[0], hop_length=hop_length, sr=self.__sr_music)


            ax[2].semilogx(freqs_record[1:], np.mean(tempogram_record[1:], axis=1),
                           label='Mean local autocorrelation', basex=2)
            ax[2].semilogx(freqs_record[1:], ac_global_record[1:], '--', alpha=0.75,
                           label='Global autocorrelation', basex=2)
            ax[2].axvline(tempo_record, color='black', linestyle='--', alpha=.8,
                          label='Estimated tempo={:g}'.format(tempo_record))
            ax[2].legend(title='Recording:',frameon=True)
            ax[2].set(xlabel='BPM')
            ax[2].grid(True)

            ax[3].semilogx(freqs_music[1:], np.mean(tempogram_music[1:], axis=1),
                           label='Mean local autocorrelation', basex=2)
            ax[3].semilogx(freqs_music[1:], ac_global_music[1:], '--', alpha=0.75,
                           label='Global autocorrelation', basex=2)
            ax[3].axvline(tempo_music, color='black', linestyle='--', alpha=.8,
                          label='Estimated tempo={:g}'.format(tempo_music))
            ax[3].legend(title='Music:',frameon=True)
            ax[3].set(xlabel='BPM')
            ax[3].grid(True)
            plt.show()
        return format(tempo_music), format(tempo_record)
    # </editor-fold>


    # <editor-fold desc="音高特征展示">
    ######音高以阶梯图来展示，返回其相似度
    def Pitch_Step_Show(self,show=True):
        chroma_cq_record = librosa.feature.chroma_cqt(y=self.__y_record, sr=self.__sr_record)
        chroma_cq_music = librosa.feature.chroma_cqt(y=self.__y_music, sr=self.__sr_music)
        music_length=min(len(chroma_cq_record[0]),len(chroma_cq_music[0]))
        pitch_record=np.zeros(music_length)
        pitch_music = np.zeros(music_length)
        Difference=0
        for i in range(0, music_length):
            for j in range(0,12):
                if chroma_cq_record[j][i]==1:
                    pitch_record[i]=j
                if chroma_cq_music[j][i]==1:
                    pitch_music[i]=j
            if pitch_record[i]!=pitch_music[i]:
                Difference+=1
        Similarity=round(1- Difference/music_length,2)#12个音符
        # x=np.arange(len(self.__y_music)/self.__sr_music)
        #Similarity


        if show==True:
            x = np.arange(music_length)
            plt.figure()
            plt.step(x=x, y=pitch_record , where='mid' ,label='recording')
            plt.step(x=x+10, y=pitch_music + 0.1, where='mid' ,label='music')
            plt.legend(title='Similarity='+str(Similarity)+'\nline:')
            plt.show()
        return Similarity
    ######音高以CQ为分界的光谱图来展示
    def Chroma_Cq_Show(self):
        chroma_cq_record = librosa.feature.chroma_cqt(y=self.__y_record, sr=self.__sr_record)
        chroma_cq_music = librosa.feature.chroma_cqt(y=self.__y_music, sr=self.__sr_music)

        plt.figure()
        plt.subplot(2, 1, 1)
        librosa.display.specshow(chroma_cq_record, y_axis='chroma', x_axis='time')
        plt.title('recording_chroma_cqt')
        plt.colorbar()

        plt.subplot(2, 1, 2)
        librosa.display.specshow(chroma_cq_music, y_axis='chroma', x_axis='time')
        plt.title('music_chroma_cqt')
        plt.colorbar()

        plt.tight_layout()
        plt.show()
    ######音高以各种光谱图来展示
    def Pitches_Specshow_Show(self):
        # 音色谱
        chroma_stft_record = librosa.feature.chroma_stft(y=self.__y_record, sr=self.__sr_record, n_chroma=12, n_fft=4096)
        # 另一种常数Q音色谱
        chroma_cq_record = librosa.feature.chroma_cqt(y=self.__y_record, sr=self.__sr_record)
        # 功率归一化音色谱
        chroma_cens_record = librosa.feature.chroma_cens(y=self.__y_record, sr=self.__sr_record)

        chroma_stft_music = librosa.feature.chroma_stft(y=self.__y_music, sr=self.__sr_music, n_chroma=12, n_fft=4096)
        # 另一种常数Q音色谱
        chroma_cq_music = librosa.feature.chroma_cqt(y=self.__y_music, sr=self.__sr_music)
        # 功率归一化音色谱
        chroma_cens_music = librosa.feature.chroma_cens(y=self.__y_music, sr=self.__sr_music)

        plt.figure(figsize=(15, 15))
        plt.subplot(3, 2, 1)
        librosa.display.specshow(chroma_stft_record, y_axis='chroma')
        plt.title('recording_chroma_stft')
        plt.colorbar()
        plt.subplot(3, 2, 3)
        librosa.display.specshow(chroma_cq_record, y_axis='chroma', x_axis='time')
        plt.title('recording_chroma_cqt')
        plt.colorbar()
        plt.subplot(3, 2, 5)
        librosa.display.specshow(chroma_cens_record, y_axis='chroma', x_axis='time')
        plt.title('recording_chroma_cens')
        plt.colorbar()

        plt.subplot(3, 2, 2)
        librosa.display.specshow(chroma_stft_music, y_axis='chroma')
        plt.title('music_chroma_stft')
        plt.colorbar()
        plt.subplot(3, 2, 4)
        librosa.display.specshow(chroma_cq_music, y_axis='chroma', x_axis='time')
        plt.title('music_chroma_cqt')
        plt.colorbar()
        plt.subplot(3, 2, 6)
        librosa.display.specshow(chroma_cens_music, y_axis='chroma', x_axis='time')
        plt.title('music_chroma_cens')
        plt.colorbar()
        plt.tight_layout()
        plt.show()
    # </editor-fold>

    # <editor-fold desc="节拍特征图展示">

    def Beat_PLP_Show(self):#主要本地脉冲PLP估计
        ###############

        fig, ax = plt.subplots(nrows=3, ncols=2, sharex=True)
        librosa.display.specshow(librosa.power_to_db(self.__melspec_record,
                                                     ref=np.max),
                                 x_axis='time', y_axis='mel', ax=ax[0][0])
        ax[0][0].set(title='Recording Mel spectrogram')
        ax[0][0].label_outer()
        ax[1][0].plot(librosa.times_like(self.__onset_env_record),
                   librosa.util.normalize(self.__onset_env_record),
                   label='Onset strength')
        ax[1][0].plot(librosa.times_like(self.__pulse_record),
                   librosa.util.normalize(self.__pulse_record),
                   label='Predominant local pulse (PLP)')
        ax[1][0].set(title='Uniform tempo prior [30, 300]')
        ax[1][0].label_outer()
        ax[2][0].plot(librosa.times_like(self.__onset_env_record),
                   librosa.util.normalize(self.__onset_env_record),
                   label='Onset strength')
        ax[2][0].plot(librosa.times_like(self.__pulse_lognorm_record),
                   librosa.util.normalize(self.__pulse_lognorm_record),
                   label='Predominant local pulse (PLP)')
        ax[2][0].set(title='Log-normal tempo prior, mean=120', xlim=[5, 20])
        ax[2][0].legend()
        ######################
        librosa.display.specshow(librosa.power_to_db(self.__melspec_music,
                                                     ref=np.max),
                                 x_axis='time', y_axis='mel', ax=ax[0][1])
        ax[0][1].set(title='Music Mel spectrogram')
        ax[0][1].label_outer()
        ax[1][1].plot(librosa.times_like(self.__onset_env_music),
                   librosa.util.normalize(self.__onset_env_music),
                   label='Onset strength')
        ax[1][1].plot(librosa.times_like(self.__pulse_music),
                   librosa.util.normalize(self.__pulse_music),
                   label='Predominant local pulse (PLP)')
        ax[1][1].set(title='Uniform tempo prior [30, 300]')
        ax[1][1].label_outer()
        ax[2][1].plot(librosa.times_like(self.__onset_env_music),
                   librosa.util.normalize(self.__onset_env_music),
                   label='Onset strength')
        ax[2][1].plot(librosa.times_like(self.__pulse_lognorm_music),
                   librosa.util.normalize(self.__pulse_lognorm_music),
                   label='Predominant local pulse (PLP)')
        ax[2][1].set(title='Log-normal tempo prior, mean=120', xlim=[5, 20])
        ax[2][1].legend()
        plt.tight_layout()
        plt.show()

    def Beat_Track_Show(self):#节拍跟踪器
        hop_length = 512
        fig, ax = plt.subplots(nrows=2,ncols=2, sharex=True)
        times = librosa.times_like(self.__onset_env_record, sr=self.__sr_record, hop_length=hop_length)
        M = librosa.feature.melspectrogram(y=self.__y_record, sr=self.__sr_record, hop_length=hop_length)
        # librosa.display.specshow(librosa.power_to_db(M, ref=np.max),
        #                          y_axis='mel', x_axis='time', hop_length=hop_length,
        #                          ax=ax[0])
        librosa.display.specshow(self.__logmelspec_record, sr=self.__sr_record, x_axis='time', y_axis='mel',ax=ax[0][0])
        ax[0][0].label_outer()
        ax[0][0].set(title='Recording Mel spectrogram')
        ax[1][0].plot(times, librosa.util.normalize(self.__onset_env_record),
                   label='Onset strength')
        ax[1][0].vlines(times[self.__beats_music], 0, 1, alpha=0.5, color='r',############
                     linestyle='--', label='Beats')
        ax[1][0].legend()

        times = librosa.times_like(self.__onset_env_music, sr=self.__sr_music, hop_length=hop_length)
        M = librosa.feature.melspectrogram(y=self.__y_music, sr=self.__sr_music, hop_length=hop_length)
        librosa.display.specshow(self.__logmelspec_music, sr=self.__sr_music, x_axis='time', y_axis='mel', ax=ax[0][1])
        ax[0][1].label_outer()
        ax[0][1].set(title='Music Mel spectrogram')
        ax[1][1].plot(times, librosa.util.normalize(self.__onset_env_music),
                   label='Onset strength')
        ax[1][1].vlines(times[self.__beats_music], 0, 1, alpha=0.5, color='r',
                     linestyle='--', label='Beats')
        ax[1][1].legend()
        plt.show()

    def Beat_Wavform(self):
        plt.figure()
        plt.subplot(2, 2, 1)
        librosa.display.waveplot(self.__y_music, self.__sr_music)
        plt.title('Music Beat wavform')

        plt.subplot(2, 2, 3)
        librosa.display.specshow(self.__logmelspec_music, sr=self.__sr_music, x_axis='time', y_axis='mel')
        plt.title('Music Mel spectrogram')
        plt.tight_layout()

        plt.subplot(2, 2, 2)
        librosa.display.waveplot(self.__y_record, self.__sr_record)
        plt.title('Recording Beat wavform')

        plt.subplot(2, 2, 4)
        librosa.display.specshow(self.__logmelspec_record, sr=self.__sr_record, x_axis='time', y_axis='mel')
        plt.title('Recording Mel spectrogram')
        plt.tight_layout()

        plt.show()
    # </editor-fold>