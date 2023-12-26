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

alias GGUFDataPart = Int
alias Parts = DynamicVector[Tensor[DType.uint32]]
alias DataParts = DynamicVector[GGUFDataPart]

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

    var types: Pointer[GGUFValueType]

struct ReaderTensor:
    var name: String
    var tensor_type: GGMLQuantizationType
    var shape : Pointer[UInt32]
    var n_elements: Int
    var n_bytes: Int
    var data_offset: Int
    var data: Pointer[AnyType]
    var field: ReaderField


struct GGUFReader:
    # I - same as host, S - swapped
    var byte_order : String
    var data : MemMapBuffer
    var fields : DynamicVector[ReaderField]
    var field_names : PythonObject

    alias alignment = GGUF_DEFAULT_ALIGNMENT

    def __init__(inout self, path: StringRef, byte_order: String = 'I'):
        let builtins = Python.import_module("builtins")

        self.data = MemMapBuffer(path)
        self.byte_order = byte_order

        self.fields = DynamicVector[ReaderField]()
        self.field_names = builtins.set()

        offs = 0
        if self.data.get[DType.uint32](offs, 1)[0] != GGUF_MAGIC:
            raise 'GGUF magic invalid'
        offs += 4
        version = self.data.get[DType.uint32](offs, 1)[0]
        if version != GGUF_VERSION:
            raise 'Sorry, file appears to be version' + String(version) + ' which we cannot handle'
        print('GGUF version ' + String(version[0]))
    
    def _push_field(self, field: ReaderField, skip_sum: Bool =  False) -> Int:
        if self.field_names.contains(field.name):
            raise 'Duplicate ' + field.name + ' already in list at offset ' + String(field.offset)
        self.fields.push_back(field)
        if skip_sum:
            return 0

        var bytes = 0
        for i in range(self.fields.capacity):
            for pi in range(self.fields[i].parts.capacity):
                bytes += 0 # self.fields[i].parts[pi].nbytes
        return bytes