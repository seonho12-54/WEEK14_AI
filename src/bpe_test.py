from bpe import BPETokenizer

corpus = "이 영화는 정말 재미있다. 이 영화는 정말 좋았다."
text = "이 영화는 정말 좋았다."

tokenizer = BPETokenizer(vocab_size=300)
tokenizer.train(corpus)

ids_before = tokenizer.encode(text, add_bos_eos=True)
decoded_before = tokenizer.decode(ids_before)

tokenizer.save("data/test_bpe_vocab.json")

loaded = BPETokenizer()
loaded.load("data/test_bpe_vocab.json")

ids_after = loaded.encode(text, add_bos_eos=True)
decoded_after = loaded.decode(ids_after)

print(ids_before)
print(decoded_before)
print(ids_after)
print(decoded_after)

print(ids_before == ids_after)
print(decoded_before == decoded_after)