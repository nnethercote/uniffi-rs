class RustString(ctypes.c_void_p):
    """Helper class for handling pointers to rust-owned strings."""

    @staticmethod
    def allocFromString(value):
        error = RustError()
        error.code = 0
        try:
            return _UniFFILib.{{ ci.ffi_string_alloc_from().name() }}(value.encode("utf8"), error)
        finally:
            if error.code != 0:
                message = str(error)
                error.free()
                raise RuntimeError(message)

    def free(self):
        return _UniFFILib.{{ ci.ffi_string_free().name() }}(self)

    def __str__(self):
        data = ctypes.cast(self, ctypes.c_char_p).value
        return "RustString(%r)".format(data)

    def consumeIntoString(self):
        try:
           return ctypes.cast(self, ctypes.c_char_p).value.decode("utf8")
        finally:
            self.free()
