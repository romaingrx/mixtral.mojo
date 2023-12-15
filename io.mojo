from types import FileBuf


fn read_file(file_name: String, inout buf: FileBuf) raises:
    var fd = open(file_name, "r")
    var data = fd.read()
    fd.close()
    buf.size = data._buffer.size
    buf.data = data._steal_ptr().bitcast[DType.uint8]()
    buf.offset = 0
    return None
