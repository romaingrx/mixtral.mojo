from sort import quicksort
from types import (
    PointerString,
    PointerStrings,
    BufferPtrFloat32,
    FileBuf,
    read_val_int,
    read_val_float32,
    read_val_str,
    string_compare,
    wrap,
    str_to_ptr,
    str_concat,
)


struct Tokenizer:
    var vocab: PointerStrings
    var vocab_scores: BufferPtrFloat32
    var max_token_length: Int
    var vocab_size: Int
    var sorted_vocab: PointerStrings
    var sorted_indices: DynamicVector[Int]

    fn __init__(inout self, vocab_size: Int, inout buf: FileBuf) raises -> None:
        self.vocab_size = vocab_size
        self.max_token_length = read_val_int(buf)
        self.vocab_scores = BufferPtrFloat32.alloc(self.vocab_size)
        self.vocab = PointerStrings.alloc(self.vocab_size)
        # lazy load sorted vocab
        self.sorted_vocab = PointerStrings.alloc(0)
        self.sorted_indices = DynamicVector[Int](0)

        # read vocab_scores & vocab values (tokens)
        for i in range(0, self.vocab_size):
            let score = read_val_float32(buf)
            let slen = read_val_int(buf)
            let token = read_val_str(buf, slen)
            self.store_token(i, token, score)
        return None

    fn __del__(owned self):
        for i in range(0, self.vocab_size):
            self.vocab[i].free()
        self.vocab.free()
        self.vocab_scores.free()
        self.sorted_vocab.free()

    fn store_token(
        inout self, index: Int, owned token: PointerString, score: Float32
    ) -> None:
        self.vocab_scores.store(index, score)
        self.vocab.store(index, token)

    # sort vocab by string_compare
    fn sort(inout self) -> None:
        if len(self.sorted_indices) < self.vocab_size:
            self.sorted_indices = DynamicVector[Int](self.vocab_size)
            self.sorted_vocab = PointerStrings.alloc(self.vocab_size)
            for ii in range(self.vocab_size):
                self.sorted_vocab.store(ii, self.vocab[ii])
                self.sorted_indices.push_back(ii)

        let n = self.vocab_size
        quicksort(self.sorted_vocab, self.sorted_indices, 0, n - 1)
        return None

    # Binary search that returns -1 if string is not found
    fn find(inout self, token_o: PointerString) -> Int:
        let token = wrap(token_o)
        let n = self.vocab_size
        if len(self.sorted_indices) < n:
            self.sort()
        var left = 0
        var right = n - 1
        while left <= right:
            let mid = left + (right - left) // 2
            let comparison = string_compare(self.sorted_vocab[mid], token)
            if comparison == 0:
                return self.sorted_indices[mid]
            if comparison < 0:
                left = mid + 1
            else:
                right = mid - 1
        return -1

    fn _bpe_encode(inout self, inout tokens: DynamicVector[Int], text: String):
        for pos in range(len(text)):
            let char = str_to_ptr(text[pos])
            let id = self.find(char)
            if id == -1:
                print("Not a good prompt token at pos ", pos)
                return
            tokens.push_back(id)

        while True:
            var best_score = Float32(-1e10)
            var best_id = -1
            var best_idx = -1

            for i in range(len(tokens) - 1):
                # Check if we can merge the pair (tokens[i], tokens[i+1])
                let str = str_concat(self.vocab[tokens[i]], self.vocab[tokens[i + 1]])
                let id = self.find(str)
                if id != -1 and self.vocab_scores.load(id) > best_score:
                    best_score = self.vocab_scores.load(id)
                    best_id = id
                    best_idx = i

            if best_idx == -1:
                # We couldn't find any more pairs to merge, so we're done
                break

            # Merge the consecutive pair (best_idx, best_idx+1) into new token best_id
            tokens[best_idx] = best_id
            # Delete token at position best_idx+1, shift the entire sequence back 1
            var _tokens = DynamicVector[Int]()
            for i in range(0, best_idx + 1):
                _tokens.push_back(tokens[i])
            for i in range(best_idx + 2, len(tokens)):
                _tokens.push_back(tokens[i])
            tokens = _tokens

    fn encode(inout self, inout tokens: DynamicVector[Int], text: String):
       return self._bpe_encode(tokens, text)