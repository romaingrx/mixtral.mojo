#
# GGUF file reading/modification support. For API usage information,
# please see the files scripts/ for some fairly simple examples.
from .constants import (
    GGML_QUANT_SIZES,
    GGUF_DEFAULT_ALIGNMENT,
    GGUF_MAGIC,
    GGUF_VERSION,
    GGMLQuantizationType,
    GGUFValueType,
)
from .io import MemMapBuffer

from python import Python
from collections.vector import DynamicVector
from memory import memset


# TODO : Create a struct that can handle multi type data for the different parts,
#        it should have a type field and then a field for each type that will be populated
#        based on the type field.

alias GGUFDataPart = Int
alias GGUFPart = Tensor[DType.uint32]
alias DataParts = DynamicVector[GGUFDataPart]
alias Parts = DynamicVector[GGUFPart]


@value
struct ReaderField(CollectionElement):
    # Offset to start of this field.
    var offset: Int

    # Name of the field (not necessarily from file data).
    var name: String

    # Data parts. Some types have multiple components, such as strings
    # that consist of a length followed by the string data.
    var parts: Parts

    # Indexes into parts that we can call the actual data. For example
    # an array of strings will be populated with indexes to the actual
    # string data.
    var data: DataParts

    var types: Pointer[Int]

    @always_inline
    def __init__(
        inout self,
        offset: Int,
        name: String,
        part: GGUFPart,
        data: GGUFDataPart,
        type: Int,
    ):
        var parts = DynamicVector[Tensor[DType.uint32]]()
        parts.push_back(part)

        var data_parts = DynamicVector[GGUFDataPart]()
        data_parts.push_back(data)

        let types = Pointer[Int].alloc(1)
        types[0] = type

        self.offset = offset
        self.name = name
        self.parts = parts
        self.types = types
        self.data = data_parts

    @always_inline
    def __init__(
        inout self,
        offset: Int,
        name: String,
        parts: Parts,
        data: DataParts,
        types: Pointer[Int],
    ):
        self.offset = offset
        self.name = name
        self.parts = parts
        self.types = types
        self.data = data


struct ReaderTensor:
    var name: String
    var tensor_type: GGMLQuantizationType
    var shape: Pointer[UInt32]
    var n_elements: Int
    var n_bytes: Int
    var data_offset: Int
    var data: Pointer[AnyType]
    var field: ReaderField


@value
struct GGUFString:
    var length: UInt64
    var data: Tensor[DType.uint8]
    var string: String


@value
struct GGUFFieldPart:
    var offset: Int
    var types: DynamicVector[Int]
    var idx: DynamicVector[Int]
    var data: DTypePointer[DType.uint8]

    @always_inline
    def __init__(
        inout self,
        offset: Int,
        types: DynamicVector[Int],
        idx: Int,
        data: DTypePointer[DType.uint8],
    ):
        self.offset = offset
        self.types = types
        var vidx = DynamicVector[Int]()
        vidx.push_back(idx)
        self.idx = vidx
        self.data = data


struct GGUFReader:
    # I - same as host, S - swapped
    var byte_order: String
    var data: MemMapBuffer
    var fields: DynamicVector[ReaderField]
    var field_names: PythonObject

    var version: UInt32
    var tensor_count: UInt64
    var kv_count: UInt64

    alias alignment = GGUF_DEFAULT_ALIGNMENT

    def __init__(inout self, path: StringRef, byte_order: String = "I"):
        let builtins = Python.import_module("builtins")

        self.data = MemMapBuffer(path)
        self.byte_order = byte_order

        self.fields = DynamicVector[ReaderField]()
        self.field_names = builtins.set()

        offs = 0
        let magic = self.data.get[DType.uint32](offs, 1)[0]
        if magic != GGUF_MAGIC:
            raise "GGUF magic invalid"
        offs += 4
        self.version = self.data.get[DType.uint32](offs, 1)[0]
        if self.version != GGUF_VERSION:
            raise "Sorry, file appears to be version" + String(
                self.version
            ) + " which we cannot handle"
        offs += 4

        self.tensor_count = self.data.get[DType.uint64](offs, 1)[0]
        offs += 8
        self.kv_count = self.data.get[DType.uint64](offs, 1)[0]
        offs += 8

        offs = self._build_fields(offs, self.kv_count.to_int())

        print("GGUF magic " + String(magic))
        print("GGUF version " + String(self.version))
        print("GGUF tensor count " + String(self.tensor_count))
        print("GGUF kv count " + String(self.kv_count))

    def _push_field(inout self, field: ReaderField, skip_sum: Bool = False) -> Int:
        let builtins = Python.import_module("builtins")
        if (self.field_names.intersection(builtins.set(field.name))).__len__() != 0:
            raise "Duplicate " + field.name + " already in list at offset " + String(
                field.offset
            )

        self.fields.push_back(field)
        self.field_names.add(field.name)
        if skip_sum:
            return 0

        var bytes = 0
        for i in range(self.fields.size):
            for pi in range(self.fields[i].parts.size):
                bytes += self.fields[i].parts[pi].bytecount()
        return bytes

    fn _get_str(self, inout offset: Int) -> GGUFString:
        let klen = self.data.get[DType.uint64](offset, 1)[0].to_int()
        offset += 8
        let kdata = self.data.get[DType.uint8](offset, klen)
        var s: String = ""
        for i in range(klen):
            s += chr(kdata[i].to_int())
        offset += klen
        return GGUFString(klen, kdata, s)

    fn _get_field_parts(
        self,
        inout orig_offs: Int,
        raw_type: Int,
    ) raises -> GGUFFieldPart:
        var types = DynamicVector[Int]()
        types.push_back(raw_type)

        if raw_type == GGUFValueType.STRING:
            print("Orig offs " + String(orig_offs))
            let offs = orig_offs
            let kv = self._get_str(orig_offs)
            print("Found string " + kv.string)
            let raw_data = self.data.get[DType.uint8](offs, 4 + kv.length.to_int())._ptr
            return GGUFFieldPart(orig_offs, types, 1, raw_data)

        try:
            let dtype = GGUFValueType.get_dtype(raw_type)
            let val = self.data.get[DType.uint8](orig_offs, dtype.sizeof())._ptr
            return GGUFFieldPart(orig_offs, types, 0, val)
        except:
            # TODO : Handle only error from get_dtype
            pass
        raise "Unknown type " + String(raw_type)

    fn _build_fields(self, inout offs: Int, count: Int) raises -> Int:
        for _ in range(count):
            let orig_offs = offs
            let kv = self._get_str(offs)
            print("KV data " + kv.string)
            let raw_kv_type = self.data.get[DType.uint32](offs, 1)[0]
            print("KV type " + String(raw_kv_type))
            offs += 4
            let field_part = self._get_field_parts(offs, raw_kv_type.to_int())
            for i in range(4, 9):
                print_no_newline(field_part.data.bitcast[DType.uint8]()[i])
            break
        return offs
