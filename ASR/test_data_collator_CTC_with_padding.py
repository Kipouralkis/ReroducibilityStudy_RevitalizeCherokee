import run_speech_recognition_ctc as asr
import unittest

import torch
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoTokenizer,
    Wav2Vec2Processor,
)

class TestDataCollatorCTCWithPadding(unittest.TestCase):
    def test_processor_call(self):

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
        padding = "longest"
        pad_to_multiple_of = None
        pad_to_multiple_of_labels = None
        
        feature = [{'input_values': [1, 2, 3], 'labels': [1, 2, 3]}]
        result = processor(feature)
        expected_result = {
            'input_values': torch.tensor([[1, 2, 3]]),
            'labels': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        self.assertEqual(result, expected_result)
        
    def test_create_vocabulary_from_data(self):
        datasets = {'train': [{'target_text': ["Sóhně:la aye̋:hli à:tlíɂili", "Galv̌:ládi:dla"]}]}
        result = asr.create_vocabulary_from_data(datasets)
        expected_result = {' ': 0, ':': 1, 'G': 2, 'S': 3, 'a': 4, 'd': 5, 'e': 6, 'h': 7, 'i': 8, 'l': 9, 'n': 10, 't': 11, 'v': 12, 'y': 13, 'à': 14, 'á': 15, 'í': 16, 'ó': 17, 'ě': 18, 'ɂ': 19, '̋': 20, '̌': 21, '[UNK]': 22, '[PAD]': 23}
        self.assertEqual(result, expected_result)

if __name__ == '__main__':
    unittest.main()
