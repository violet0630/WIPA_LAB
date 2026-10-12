"""阶段3B（任务3-2）：课件中的 Front Coding 与 Gap + Variable-Byte 通用编解码器。

本文件是纯编解码库，被 03_compress_index.py（编码）、03_verify_compression.py 与
04A_boolean_retrieval.py（解码检索）共同复用；不直接运行。

使用约定：
- 字典词项为UTF-8字节串；Front Coding 每个块的第一个词存全文，其余词存
  与前一个词的公共前缀字节数 + 剩余后缀。
- 变长整数使用课件约定：每字节低7位存有效数据，最高位1表示最后一个字节，
  最低的7位先输出（little-endian 7-bit chunks）。
- posting 编码为正 doc gap、title位置数、title位置gap、content位置数、
  content位置gap。位置按字段独立从 -1 开始计算gap，因此没有0或负的gap。
"""
from __future__ import annotations

import io
import struct

U32 = struct.Struct('<I')
# offset: u64; length: u64; df: u32; skip_step: u32 = 24 bytes per term
ENTRY = struct.Struct('<QQII')


def vbyte_write(value: int, output: bytearray) -> None:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f'VByte只能编码非负整数：{value!r}')
    while value >= 128:
        output.append(value & 0x7F)  # highest bit 0: not final
        value >>= 7
    output.append((value & 0x7F) | 0x80)  # highest bit 1: final


def vbyte_read(stream) -> int:
    value = 0
    shift = 0
    while True:
        b = stream.read(1)
        if not b:
            raise EOFError('截断的 VByte 整数')
        code = b[0]
        value |= (code & 0x7F) << shift
        if code & 0x80:
            return value
        shift += 7
        if shift > 63:
            raise ValueError('VByte整数过长（超过64位）')


def enc_length(n: int, out: bytearray) -> None:
    # 词典长度是序列化元数据，所有词典变体统一使用相同的长度表示规则。
    vbyte_write(n, out)


def dec_length(stream) -> int:
    return vbyte_read(stream)


def common_prefix_len(a: bytes, b: bytes) -> int:
    i = 0
    for x, y in zip(a, b):
        if x != y:
            break
        i += 1
    return i


class DictionaryEncoder:
    def __init__(self, mode: str, block_size: int = 8):
        if mode not in ('plain', 'front'):
            raise ValueError('mode必须为plain或front')
        if block_size < 2:
            raise ValueError('block_size必须>=2')
        self.mode = mode
        self.block_size = block_size
        self.previous = b''
        self.count = 0

    def encode(self, term: str) -> bytes:
        raw = term.encode('utf-8')
        out = bytearray()
        if self.mode == 'front' and self.count % self.block_size:
            prefix = common_prefix_len(self.previous, raw)
            suffix = raw[prefix:]
            enc_length(prefix, out)
            enc_length(len(suffix), out)
            out.extend(suffix)
        else:
            enc_length(len(raw), out)
            out.extend(raw)
        self.previous = raw
        self.count += 1
        return bytes(out)


class DictionaryDecoder:
    def __init__(self, mode: str, block_size: int = 8):
        self.mode = mode
        self.block_size = block_size
        self.previous = b''
        self.count = 0

    def decode(self, handle) -> str:
        if self.mode == 'front' and self.count % self.block_size:
            prefix = dec_length(handle)
            suffix_len = dec_length(handle)
            if prefix > len(self.previous):
                raise ValueError('字典Front Coding前缀长度超出上一词长度')
            suffix = handle.read(suffix_len)
            if len(suffix) != suffix_len:
                raise EOFError('字典后缀截断')
            raw = self.previous[:prefix] + suffix
        else:
            length = dec_length(handle)
            raw = handle.read(length)
            if len(raw) != length:
                raise EOFError('字典词项截断')
        self.previous = raw
        self.count += 1
        return raw.decode('utf-8')


def _put_integer(n: int, out: bytearray, vbyte: bool) -> None:
    if vbyte:
        vbyte_write(n, out)
    else:
        if n < 0 or n >= 2**32:
            raise ValueError('固定32位存储溢出')
        out.extend(U32.pack(n))


def _get_integer(stream, vbyte: bool) -> int:
    if vbyte:
        return vbyte_read(stream)
    b = stream.read(4)
    if len(b) != 4:
        raise EOFError('32位整数读取不完整')
    return U32.unpack(b)[0]


def encode_postings(postings: list, variable_byte: bool) -> bytes:
    result = bytearray()
    previous_doc = -1
    for record in postings:
        if not isinstance(record, list) or len(record) != 3:
            raise ValueError('posting应为[doc_num,title_positions,content_positions]')
        doc, titles, contents = record
        if not isinstance(doc, int) or doc <= previous_doc:
            raise ValueError('doc_num必须严格递增')
        if not titles and not contents:
            raise ValueError('空posting不能编码')
        _put_integer(doc - previous_doc, result, variable_byte)
        previous_doc = doc
        for positions in (titles, contents):
            _put_integer(len(positions), result, variable_byte)
            previous_pos = -1
            for pos in positions:
                if not isinstance(pos, int) or pos <= previous_pos:
                    raise ValueError('同一字段的位置信息必须严格递增')
                _put_integer(pos - previous_pos, result, variable_byte)
                previous_pos = pos
    return bytes(result)


def decode_postings(payload: bytes, df: int, variable_byte: bool) -> list:
    stream = io.BytesIO(payload)
    result = []
    previous_doc = -1
    for _ in range(df):
        gap = _get_integer(stream, variable_byte)
        if gap < 1:
            raise ValueError('文档差值必须为正整数')
        doc = previous_doc + gap
        previous_doc = doc
        parts = []
        for _field in range(2):
            count = _get_integer(stream, variable_byte)
            if count > len(payload):
                raise ValueError('无效的位置个数')
            positions = []
            prev_pos = -1
            for _i in range(count):
                diff = _get_integer(stream, variable_byte)
                if diff < 1:
                    raise ValueError('位置差值必须为正整数')
                prev_pos += diff
                positions.append(prev_pos)
            parts.append(positions)
        if not parts[0] and not parts[1]:
            raise ValueError('空posting')
        result.append([doc, parts[0], parts[1]])
    if stream.read(1):
        raise ValueError('存在多余的posting字节')
    return result
