import torch
import torchaudio
from torch.utils.data import Dataset
import run_speech_recognition_ctc as asr

import torch
import torchaudio
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoTokenizer,
    Wav2Vec2Processor,
)


class AudioDataset(Dataset):
    def __init__(self, audio_files, target_texts):
        self.audio_files = audio_files
        self.target_texts = target_texts
    def __getitem__(self, index):
        return {
            "data_args.audio_column_name": self.audio_files[index],
            "target_text": self.target_texts[index]
        }
    def __len__(self):
        return len(self.audio_files)

def test_prepare_dataset(processor, resampler):
    # Create a real audio dataset
    audio_files = [
        "chr_voice\clips\cno_cno_cwl_Teeth.mp3",
        "chr_voice\clips\walc-1_mp3_1073-Chapter_8_8.5_8.5.16-000000000.mp3",
        "chr_voice\clips\durbin-feeling-tones_mp3_0105-df-tones_enhanced-290465.mp3"
    ]
    target_texts = [
        "de:gánhdóhgv̋",
        "Kǎ:hwi",
        "À:da:náɂnv̋:ɂi"
    ]
    raw_datasets = {
        "train": AudioDataset(audio_files, target_texts)
    }

    # set max_input_length
    max_input_length = 10
    min_input_length = 0.0

    # Test prepare_dataset
    vectorized_datasets = raw_datasets["train"].map(
        asr.prepare_dataset,
        remove_columns=raw_datasets["train"].column_names,
        num_proc=1,
    )
    for data in vectorized_datasets:
        # check if input_values is correctly processed
        assert "input_values" in data
        assert data["input_values"].shape[0] <= max_input_length
        # check if labels is correctly processed
        assert "labels" in data
        assert isinstance(data["labels"], torch.Tensor)

    # Test prepare_dataset with min_input_length
    vectorized_datasets = raw_datasets["train"].map(
        asr.prepare_dataset,
        remove_columns=raw_datasets["train"].column_names,
        num_proc=1,
    ).filter(
        lambda data: len(data["input_values"]) > min_input_length,
    )
    for data in vectorized_datasets:
        # check if input_values is correctly filtered
        assert len(data["input_values"]) > min_input_length

if __name__ == '__main__':

    config = AutoConfig.from_pretrained("facebook/wav2vec2-large-xlsr-53")

    # load feature_extractor, tokenizer and create processor
    tokenizer = AutoTokenizer.from_pretrained(
        "wav2vec2-chr_voice-ep15-lr0.0003-mask0.01",
        tokenizer_type=config.model_type,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        "facebook/wav2vec2-large-xlsr-53"
    )

    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    resampler = torchaudio.transforms.Resample(48_000, processor.feature_extractor.sampling_rate)

    test_prepare_dataset(processor, resampler)
