# -*- coding: utf-8 -*-
"""
UTF-8 byte-level BPE 토크나이저 과제 템플릿.

외부 tokenizer 라이브러리 없이 BPE(Byte Pair Encoding)를 직접 구현합니다.
한국어 NSMC 리뷰를 다루므로 문자열을 글자/공백 단위로 먼저 자르지 말고,
항상 `text.encode("utf-8")`로 byte ID 시퀀스를 만든 뒤 merge를 적용하세요.
"""

from pathlib import Path
from collections import Counter
import json

# 문장마다 길이가 다르기 때문에, batch로 묶으려면 길이를 맞춰야함. 
# 문장 A: [나는, 밥을, 먹었다]
# 문장 B: [좋다, <pad>, <pad>]
PAD_TOKEN = "<pad>"

# tokenizer가 모르는 글자나 토큰을 만났을 때 대신 쓰는 토큰
UNK_TOKEN = "<unk>"

# beginning of sequence 문장의 시작을 표시
BOS_TOKEN = "<bos>"

# end of sequence 문장의 끝을 표시
EOS_TOKEN = "<eos>"


SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
SPECIAL_IDS = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
BYTE_OFFSET = len(SPECIAL_TOKENS)
NUM_BYTES = 256


class BPETokenizer:
    """
    UTF-8 byte-level BPE 토크나이저.

    권장 ID 배치:
    - 0~3: <pad>, <unk>, <bos>, <eos>
    - 4~259: 원본 byte 0~255
    - 260 이상: BPE merge로 생성한 토큰
    """

    def __init__(self, vocab_size: int = 3000):
        self.vocab_size = vocab_size
        self.id_to_token = {}
        self.token_to_id = {}
        self.merges = []

    def _init_special_tokens(self):
        """
        TODO:
        1. 특수 토큰 4개를 고정 ID 0~3에 등록합니다.
        2. byte 0~255를 ID 4~259에 bytes([byte_value]) 형태로 등록합니다.
        """
        for item in SPECIAL_TOKENS:
            self.token_to_id[item] = SPECIAL_IDS[item]
            self.id_to_token[SPECIAL_IDS[item]] = item
        
        for i in range(NUM_BYTES):
            self.token_to_id[i] = i + BYTE_OFFSET
            self.id_to_token[i + BYTE_OFFSET] = i

    def get_pad_id(self):
        """padding 토큰 ID."""
        return SPECIAL_IDS[PAD_TOKEN]

    def get_unk_id(self):
        """unknown 토큰 ID."""
        return SPECIAL_IDS[UNK_TOKEN]

    def get_bos_id(self):
        """문장 시작 토큰 ID."""
        return SPECIAL_IDS[BOS_TOKEN]

    def get_eos_id(self):
        """문장 끝 토큰 ID."""
        return SPECIAL_IDS[EOS_TOKEN]

    def get_pair(self,token_id_list):
        pairs = []
        for i in range(len(token_id_list) - 1):
            pair = (token_id_list[i], token_id_list[i+1])
            pairs.append(pair)
        
        return Counter(pairs)
    
    def get_token_from_byte(self, bytes):

        result = bytes.encode("utf-8")
        token_list = []

        for item in result:
            token_list.append(self.token_to_id[item])
        
        return token_list

    def train(self, corpus: str):
        self._init_special_tokens()
        
        token_id_list = self.get_token_from_byte(corpus)

        while(True):

            pair_counts = self.get_pair(token_id_list)

            if (len(pair_counts) == 0): return 

            best_pair, count = pair_counts.most_common(1)[0]
            if(count < 2): return

            token_to_id_len = self.token_to_id.get(best_pair)

            if(token_to_id_len == None):
                token_to_id_len = len(self.token_to_id)

            first_el = best_pair[0]
            second_el = best_pair[1]
            new_token_id_list = []
            i = 0
            
            while(i < len(token_id_list)):
                if(i+1 < len(token_id_list) and token_id_list[i] == first_el and token_id_list[i + 1] == second_el):
                    new_token_id_list.append(token_to_id_len)
                    i+=2
                    continue
                new_token_id_list.append(token_id_list[i])
                i+=1

            self.token_to_id[best_pair] = token_to_id_len
            self.id_to_token[token_to_id_len] = best_pair
            self.merges.append(best_pair)
            
            if(self.vocab_size <= len(self.token_to_id)): return
            token_id_list = new_token_id_list
            """
        TODO: 코퍼스에서 BPE merge rule과 vocabulary를 학습합니다.
        구현 힌트:
        - `corpus.encode("utf-8")`로 byte ID 시퀀스를 만듭니다.
        - 가장 자주 등장하는 이웃 token pair를 찾습니다.
        - 새 token ID를 만들고, 시퀀스의 해당 pair를 새 ID로 치환합니다.
        - `self.merges`, `self.id_to_token`, `self.token_to_id`를 갱신합니다.
        """


    def save(self, path: str | Path):

        data = {
            "vocab_size": self.vocab_size,
            "id_to_token": {},
            "merges": [],
        }

        for token_id, token in self.id_to_token.items():
            if isinstance(token, str):
                token_data = {
                    "type": "str",
                    "value": token,
                }
            elif isinstance(token, int):
                token_data = {
                    "type": "int",
                    "value": token,
                }
            else:
                token_data = {
                    "type": "tuple",
                    "value": list(token),
                }
            data["id_to_token"][str(token_id)] =token_data
        
        for first_id, second_id in self.merges:
            data["merges"].append([first_id, second_id])

        path = Path(path)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        """
        TODO: vocabulary와 merge rule을 JSON 파일로 저장합니다.

        bytes와 tuple은 JSON에 바로 저장할 수 없으므로 type 정보를 함께 저장하세요.
        """
        

    def load(self, path: str | Path):
        """
        TODO: save()로 저장한 JSON 파일을 읽어 vocabulary와 merge rule을 복원합니다.
        """
        
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.vocab_size = data["vocab_size"]
        self.id_to_token = {}
        self.token_to_id = {}
        self.merges = []

        for token_id_str, token_data in data["id_to_token"].items():
            token_id = int(token_id_str)
            token_type = token_data["type"]
            value = token_data["value"]

            if token_type == "str":
                token = value
            elif token_type == "int":
                token = int(value)
            elif token_type == "tuple":
                token = tuple(value)
            
            self.id_to_token[token_id] = token
            self.token_to_id[token] = token_id
        
        for pair in data["merges"]:
            self.merges.append(tuple(pair))

    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        
        id_list = self.get_token_from_byte(text)
        
        for pair in self.merges:
            merged_id = self.token_to_id[pair]

            new_id_list = []
            i = 0

            while i < len(id_list):
                if (
                    i + 1 < len(id_list)
                    and id_list[i] == pair[0]
                    and id_list[i+1] == pair[1]
                ):
                    new_id_list.append(merged_id)
                    i+=2
                else:
                    new_id_list.append(id_list[i])
                    i+=1
            id_list = new_id_list


        if(add_bos_eos):
            id_list.insert(0,self.token_to_id[BOS_TOKEN])
            id_list.append(self.token_to_id[EOS_TOKEN])
        """
        TODO: 문자열을 token ID 리스트로 변환합니다.
        
        구현 힌트:
        - 먼저 UTF-8 byte ID 리스트를 만듭니다.
        - train/load에서 얻은 merge rule을 학습 순서대로 적용합니다.
        - add_bos_eos=True이면 앞뒤에 bos/eos ID를 붙입니다.
        """
        return id_list

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        
        byte_values = []

        def expand_token(token_id):
            token = self.id_to_token.get(token_id)

            if token is None:
                return []
            
            if isinstance(token, str):
                if skip_special:
                    return []

                return list(token.encode("utf-8"))

            if isinstance(token, int):
                return [token]

            left_id, right_id = token
            return expand_token(left_id) + expand_token(right_id)

        for token_id in ids:
            byte_values.extend(expand_token(token_id))
        
        return bytes(byte_values).decode("utf-8", errors="replace")
        """
        TODO: token ID 리스트를 문자열로 복원합니다.
        
        주의:
        - merge token은 원본 byte token까지 재귀적으로 펼칩니다.
        - byte를 하나씩 decode하지 말고, 마지막에 `bytes(...).decode("utf-8")`를 한 번만 호출합니다.
        """
