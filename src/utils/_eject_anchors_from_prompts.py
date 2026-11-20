from pathlib import Path
from hashlib import md5
import json

"""
Задача
    из raw_stability.json что имеет
        prompt,k,N,k_over_N
    вычленить из prompt anchor
    и записать в hashes hash(anchor)

"""

def eject_anchors_from_promppts(dataset_path,hash_path):
    with dataset_path.open("r",encoding="utf-8") as f, hash_path.open("w",encoding="utf-8") as h:
        for line in f:
            data = json.loads(line)
            prompt = data["prompt"]
            before_anchor_string = "Now analyze this text:\nText: "
            after_anchor_string = "\nLabel:"
            before_anchor_string_idx = prompt.find(before_anchor_string)+len(before_anchor_string)
            after_anchor_string_idx = len(prompt)-len(after_anchor_string)
            anchor = prompt[before_anchor_string_idx:after_anchor_string_idx]
            hash = md5(anchor.encode(),usedforsecurity=False)
            hash_string = hash.hexdigest()+"\n"
            h.write(hash_string)
            h.flush()
