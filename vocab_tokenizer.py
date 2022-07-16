#%%
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

examples = [
        "The quick BROWN fox jumps over the: lazy dog",
        "A pangram is a sentence that contains all the letters of the alphabet!",
        "Time to tokenize the sentences using this program!"]

tokenizer = get_tokenizer('basic_english')

def yield_tokens():
    for example in examples:
        tokens = tokenizer(example)
        print('tokens:', tokens)
        yield tokens

token_generator = yield_tokens()

vocab = build_vocab_from_iterator(token_generator)

vocab.get_stoi()

#check the index of a word
vocab['the']


# %%
