from transformers import BertTokenizer


class Tokenizer:
    def __init__(self, type='bert-base-multilingual-uncased', max_length=100, device='cpu'):
        self.instance = BertTokenizer.from_pretrained(type)
        self.instance.add_tokens([f"[2{lang}]" for lang in ["en", "it", "es", "de", "fr"]])
        self.vocab_size = len(self.instance)
        self.max_length = max_length
        self.device = device
        self.pad_index = self.convert_tokens_to_ids([self.instance.pad_token])[0]

    def __call__(self, src):
        """
        String to ids
        """
        return self.instance(
            src,
            return_tensors="pt",
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        ).to(self.device).data["input_ids"]

    def convert_tokens_to_ids(self, src):
        return self.instance.convert_tokens_to_ids(src)

    def convert_ids_to_tokens(self, src):
        return self.instance.convert_ids_to_tokens(src)

    def convert_tokens_to_string(self, src):
        return self.instance.convert_tokens_to_string(src)

    def tokenize(self, src):
        """
        String to tokens
        """
        return self.instance.tokenize(src)
