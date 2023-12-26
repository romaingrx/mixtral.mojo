from sys import ffi
from memory import unsafe, memcpy
from collections.vector import DynamicVector


struct MapOpt:
    alias MAP_SHARED = 0x01
    alias MAP_PRIVATE = 0x02


struct Prot:
    alias PROT_NONE = 0x0
    alias PROT_READ = 0x1
    alias PROT_WRITE = 0x2
    alias PROT_EXEC = 0x4


alias c_void = UInt8

alias mmap_type = fn (
    addr: Pointer[c_void],
    len: Int64,
    prot: Int32,
    flags: Int32,
    fildes: Int32,
    offset: Int64,
) -> Pointer[c_void]


struct MemMapBuffer:
    var data: DTypePointer[DType.uint8]
    var len: Int
    var prot: Int32
    var flags: Int32
    var fildes: Int32
    var offset: Int

    def __init__(inout self, fnm: StringRef):
        let handle: ffi.DLHandle
        if ffi.os_is_linux():
            handle = ffi.DLHandle("")
        # elif ffi.os_is_windows():
        # bug: if this section is un-commented, then `h`
        #      is considered uninitialized below
        #    raise "Not yet supported on Windows"
        else:
            # we just need _a_ dylib in the image
            handle = ffi.DLHandle("libate.dylib")

        let c_mmap = handle.get_function[mmap_type]("mmap")

        let fd = external_call["open", Int, Pointer[Int8], Int](
            fnm.data._as_scalar_pointer(), 0x0
        )
        if fd == -1:
            raise "Failed to open file"
        let NULL = unsafe.bitcast[c_void](0x0)

        self.data = c_mmap(NULL, 16, Prot.PROT_READ, MapOpt.MAP_SHARED, fd, 0)
        self.len = 16
        self.prot = Prot.PROT_READ
        self.flags = MapOpt.MAP_SHARED
        self.fildes = fd
        self.offset = 0

    fn _get_bitcast[T: DType](self, offset: Int) -> DTypePointer[T]:
        # Offset the data pointer to the desired location
        let offset_data = self.data.offset(offset)
        # Cast the pointer to the desired data type
        let typed_data = offset_data.bitcast[T]()
        return typed_data

    fn get[T: DType](self, offset: Int, count: Int) -> Tensor[T]:
        let typed_data = self._get_bitcast[T](offset)

        # Make a copy of the data
        let res = DTypePointer[T].alloc(count)
        memcpy[T](res, typed_data, count)

        return Tensor[T](res, count)