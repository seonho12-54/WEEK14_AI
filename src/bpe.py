# -*- coding: utf-8 -*-
"""
UTF-8 byte-level BPE 토크나이저 과제 템플릿.

외부 tokenizer 라이브러리 없이 BPE(Byte Pair Encoding)를 직접 구현합니다.
한국어 NSMC 리뷰를 다루므로 문자열을 글자/공백 단위로 먼저 자르지 말고,
항상 `text.encode("utf-8")`로 byte ID 시퀀스를 만든 뒤 merge를 적용하세요.
"""

from pathlib import Path
import json

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
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
        self.id_to_token = {}   #key : 0~255    value : "<pad>", b"\x42" 
        self.token_to_id = {}   #key : "<pad>", b"\x01"  value : 0~255
        self.merges = []

    def _init_special_tokens(self):
        """
        TODO:
        1. 특수 토큰 4개를 고정 ID 0~3에 등록합니다.
        2. byte 0~255를 ID 4~259에 bytes([byte_value]) 형태로 등록합니다.
        """
        #raise NotImplementedError("_init_special_tokens를 구현하세요.")
        #위에 스페셜 토큰 넣는거 있는데 // 실제로 스페셜토큰 id가 0~3으로 정해져있음
        #bytes([byte_value])는 튜플? 어떻게 생긴 거지? 생긴거를 안다면 0~3까지 등록할 수 있을 듯
        #id_to_token, token_to_id 딕셔너리에 저장하라는 뜻
        self.id_to_token[self.get_pad_id()] = PAD_TOKEN
        self.token_to_id[PAD_TOKEN] = self.get_pad_id()

        self.id_to_token[self.get_unk_id()] = UNK_TOKEN
        self.token_to_id[UNK_TOKEN] = self.get_unk_id()

        self.id_to_token[self.get_bos_id()] = BOS_TOKEN
        self.token_to_id[BOS_TOKEN] = self.get_bos_id()

        self.id_to_token[self.get_eos_id()] = EOS_TOKEN
        self.token_to_id[EOS_TOKEN] = self.get_eos_id()
        
        #나머지 4~255번 id에 bytes([byte_value]) 등록
        for i in range(0, NUM_BYTES):
            self.token_to_id[bytes([i])] = i +BYTE_OFFSET
            self.id_to_token[i + BYTE_OFFSET] = bytes([i])

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

    def train(self, corpus: str):
        """
        TODO: 코퍼스에서 BPE merge rule과 vocabulary를 학습합니다.

        구현 힌트:
        - `corpus.encode("utf-8")`로 byte ID 시퀀스를 만듭니다.
        - 가장 자주 등장하는 이웃 token pair를 찾습니다.
        - 새 token ID를 만들고, 시퀀스의 해당 pair를 새 ID로 치환합니다.
        - `self.merges`, `self.id_to_token`, `self.token_to_id`를 갱신합니다.
        """
        #raise NotImplementedError("BPETokenizer.train을 구현하세요.")
        self._init_special_tokens()
        #corpus를 byte타입으로 변환하고, 리스트로 감싸서 정수 타입으로 변경
        encoded = list(corpus.encode("utf-8"))
        #특수토큰4개를 고려해 4를 더한 값들로 변경
        tokens = [byte_value + BYTE_OFFSET for byte_value in encoded]
        #vocab사이즈 만큼만 저장. 3000개
        while len(self.id_to_token) < self.vocab_size:
            pair_count = {}

            for i in range(len(tokens)-1):
                pair = (tokens[i], tokens[i+1])
                if pair not in pair_count:
                    pair_count[pair] = 0
                
                pair_count[pair] += 1
            #더이상 merge할 pair가 없으면 종료
            if not pair_count:
                break
            #최대 빈도 pair 선택
            max_pair = max(pair_count, key = pair_count.get)
            #새 id 부여, byte타입으로 합치기
            new_id = len(self.id_to_token)
            left_token = self.id_to_token[max_pair[0]]
            rignt_token = self.id_to_token[max_pair[1]]
            new_token = left_token + rignt_token
            #vocab에 등록
            self.id_to_token[new_id] = new_token
            self.token_to_id[new_token] = new_id
            #merge 규칙 저장
            self.merges.append((max_pair, new_id))

            #기존 tokens에 max_pair를 new_id로 치환하기 위한 새 리스트
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == max_pair:
                    new_tokens.append(new_id)
                    i+=2
                else:
                    new_tokens.append(tokens[i])
                    i+=1
            tokens = new_tokens
        




    def save(self, path: str | Path):
        """
        TODO: vocabulary와 merge rule을 JSON 파일로 저장합니다.

        bytes와 tuple은 JSON에 바로 저장할 수 없으므로 type 정보를 함께 저장하세요.
        """
        #raise NotImplementedError("BPETokenizer.save를 구현하세요.")
        path = Path(path)
        #폴더가 없다면 생성하기
        path.parent.mkdir(parents=True, exist_ok=True)
        #id_to_token, merges를 저장해야함, byte타입을 어떻게 처리할지 >> .hex()
        tokens_data = []
        for key, value in self.id_to_token.items():
            if isinstance(value, bytes):
                tokens_data.append({
                    "id" : key,
                    "type" : "bytes",
                    "value" : value.hex()
                })
            else:
                tokens_data.append({
                    "id" : key,
                    "type" : "str",
                    "value" : value
                })
        
        #merges의 튜플을 json에 넣기 위해 리스트 형태로 변환
        merges_data = []
        for merge in self.merges:
            pair = list(merge[0])
            new_id = merge[1]
            merges_data.append({"pair" : pair, "new_id" : new_id,})
        #json에 저장할 내용들
        data = {
            "vocab_size" : self.vocab_size,
            "id_to_token" : tokens_data,
            "merges" : merges_data
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str | Path):
        """
        TODO: save()로 저장한 JSON 파일을 읽어 vocabulary와 merge rule을 복원합니다.
        """
        #raise NotImplementedError("BPETokenizer.load를 구현하세요.")
        #저장한 json 파일 열어서 데이터 꺼내기
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        #json파일에 넣으면서 타입이 변환 되었을 수도 있기에 int로 한 번 더
        self.vocab_size = int(data["vocab_size"])
        #token의 타입을 다시 bytes로
        self.id_to_token = {}
        self.token_to_id = {}
        for item in data["id_to_token"]:
            token_id = int(item["id"])
            token_type = item["type"]
            value = item["value"]

            if token_type == "bytes":
                token = bytes.fromhex(value)
            else:
                token = value

            self.id_to_token[token_id] = token
            self.token_to_id[token] = token_id
        #기존의 merges처럼 다시 튜플 타입으로 변경
        self.merges = []
        for item in data["merges"]:
            pair = tuple(item["pair"])
            pair_id = int(item["new_id"])

            self.merges.append((pair, pair_id))



    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        """
        TODO: 문자열을 token ID 리스트로 변환합니다.

        구현 힌트:
        - 먼저 UTF-8 byte ID 리스트를 만듭니다.
        - train/load에서 얻은 merge rule을 학습 순서대로 적용합니다.
        - add_bos_eos=True이면 앞뒤에 bos/eos ID를 붙입니다.
        """
        #raise NotImplementedError("BPETokenizer.encode를 구현하세요.")
        tokens = text.encode("utf-8")
        token_ids = []
        #token들의 타입을 bytes로 바꿔주고, token_to_id에 이미 있다면 ids에 추가, 없다면 unk로 추가
        for token in tokens:
            byte_token = bytes([token])
            if byte_token in self.token_to_id:
                token_ids.append(self.token_to_id[byte_token])
            else:
                token_ids.append(self.get_unk_id())
        
        #merge rule을 적용
        for pair, new_id in self.merges:
            new_token_ids = []
            i = 0
            while i < len(token_ids):
                if i < (len(token_ids) - 1) and (token_ids[i], token_ids[i+1]) == pair:
                    new_token_ids.append(new_id)
                    i += 2
                
                else:
                    new_token_ids.append(token_ids[i])
                    i += 1
            
            token_ids = new_token_ids

        #필요하다면 BOS, EOS도 추가
        if add_bos_eos:
            token_ids = [self.get_bos_id()] + token_ids + [self.get_eos_id()]

        return token_ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """
        TODO: token ID 리스트를 문자열로 복원합니다.

        주의:
        - merge token은 원본 byte token까지 재귀적으로 펼칩니다.
        - byte를 하나씩 decode하지 말고, 마지막에 `bytes(...).decode("utf-8")`를 한 번만 호출합니다.
        """
        #raise NotImplementedError("BPETokenizer.decode를 구현하세요.")
        #bytes타입으로 바꾼 token들을 모아둘 리스트
        byte_tokens = []
        #입력받은 ids리스트를 순회하면서 bytes로 변경해주기
        for token_id in ids:
            #token_id가 vocab에 없다면 <unk>로 대체하기
            if token_id not in self.id_to_token:
                byte_tokens.append(UNK_TOKEN.encode("utf-8"))
                continue
            
            token = self.id_to_token[token_id]
            #str 토큰은 특수 토큰, skip_special에 따라 건너뛰거나 bytes로 변환
            if isinstance(token, str):
                if skip_special:
                    continue
                else:
                    byte_tokens.append(token.encode("utf-8"))
                
            else:
                byte_tokens.append(token)
        #decode 한 번에 할 수 있도록 합치기
        merged = b"".join(byte_tokens)
        #decode중 해석할 수 없는 byte가 있다면 대체 문자로 바꾸도록
        text = merged.decode("utf-8", errors="replace")
        return text
        
