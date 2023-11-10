import torchaudio

file = '0_0_0_0_1_1_1_1.wav'

waveform, sample_rate = torchaudio.load(file)

encode = torchaudio.transforms.MuLawEncoding()(waveform)

decode = torchaudio.transforms.MuLawDecoding()(encode)

err = ((waveform - decode).abs() / waveform.abs()).median()
