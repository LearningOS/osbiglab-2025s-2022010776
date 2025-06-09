from datasets import load_dataset


ds = load_dataset("Locutusque/UltraTextbooks", split="train", streaming=False)

length = len(ds)
print(length)
ds = ds.select(range(int(length * 0.9)), keep_in_memory=True)
print(len(ds))

ds = ds.map(lambda x: {"length": len(x["text"])}, num_proc=64, load_from_cache_file=False)

print("saving")
ds.save_to_disk("./data/UltraTextbooks_length", num_proc=64, num_shards=64)
