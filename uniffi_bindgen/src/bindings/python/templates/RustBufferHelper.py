# Helpers for reading/writing types from/to a bytebuffer.

class RustBufferStream(object):

    def __init__(self, rbuf):
        self.rbuf = rbuf
        self.offset = 0

    def remaining(self):
        return self.rbuf.len - self.offset

    def _unpack_from(self, size, format):
        if self.offset + size > self.rbuf.len:
            raise RuntimeError("read past end of rust buffer")
        value = struct.unpack(format, self.rbuf.data[self.offset:self.offset+size])[0]
        self.offset += size
        return value

    def _pack_into(self, size, format, value):
        if self.offset + size > self.rbuf.len:
            raise RuntimeError("write past end of rust buffer")
        # XXX TODO: I feel like I should be able to use `struct.pack_into` here but can't figure it out.
        for i, byte in enumerate(struct.pack(format, value)):
            self.rbuf.data[self.offset + i] = byte
        self.offset += size

    def _read_bytes(self, size):
        if self.offset + size > self.rbuf.len:
            raise RuntimeError("read past end of rust buffer")
        data = self.rbuf.data[self.offset:self.offset+size]
        self.offset += size
        return data

    def _write_bytes(self, data):
        if self.offset + len(data) > self.rbuf.len:
            raise RuntimeError("write past end of rust buffer")
        # XXX TODO: surely there is a better way...?!
        for i, byte in enumerate(data):
            self.rbuf.data[self.offset + i] = byte
        self.offset += len(data)

    # For every type used in the interface, we provide helper methods for conveniently
    # reading and writing values of that type in a buffer. Putting them on this internal
    # helper object (rather than, say, as methods on the public classes) makes it easier
    # for us to hide these implementation details from consumers, in the face of python's
    # free-for-all type system.

    {% for typ in ci.iter_types() %}
    {% let type_name = typ.canonical_name() %}
    {%- match typ -%}

    {% when Type::Int8 -%}
    # The primitive signed 8-bit integer type.

    def readI8(self):
        return self._unpack_from(1, ">b")

    def writeI8(self, v):
        self._pack_into(1, ">b", v)

    {% when Type::UInt8 -%}
    # The primitive unsigned 8-bit integer type.

    def readU8(self):
        return self._unpack_from(1, ">B")

    def writeU8(self, v):
        self._pack_into(1, ">B", v)

    {% when Type::Int16 -%}
    # The primitive signed 16-bit integer type.

    def readI16(self):
        return self._unpack_from(2, ">h")

    def writeI16(self, v):
        self._pack_into(2, ">h", v)

    {% when Type::UInt16 -%}
    # The primitive unsigned 16-bit integer type.

    def readU16(self):
        return self._unpack_from(1, ">H")

    def writeU16(self, v):
        self._pack_into(1, ">H", v)

    {% when Type::Int32 -%}
    # The primitive signed 32-bit integer type.

    def readI32(self):
        return self._unpack_from(4, ">i")

    def writeI32(self, v):
        self._pack_into(4, ">i", v)

    {% when Type::UInt32 -%}
    # The primitive unsigned 32-bit integer type.

    def readU32(self):
        return self._unpack_from(4, ">I")

    def writeU32(self, v):
        self._pack_into(4, ">I", v)

    {% when Type::Int64 -%}
    # The primitive signed 64-bit integer type.

    def readI64(self):
        return self._unpack_from(8, ">q")

    def writeI64(self, v):
        self._pack_into(8, ">q", v)

    {% when Type::UInt64 -%}
    # The primitive unsigned 64-bit integer type.

    def readU64(self):
        return self._unpack_from(8, ">Q")

    def writeU64(self, v):
        self._pack_into(8, ">Q", v)

    {% when Type::Float32 -%}
    # The primitive 32-bit floating-point type.

    def readF32(self):
        v = self._unpack_from(4, ">f")
        return v

    def writeF32(self, v):
        self._pack_into(4, ">f", v)

    {% when Type::Float64 -%}
    # The primitive 64-bit floating-point integer type.

    def readF64(self):
        return self._unpack_from(8, ">d")

    def writeF64(self, v):
        self._pack_into(8, ">d", v)

    {% when Type::Boolean -%}
    # The primitive boolean type.

    def readBool(self):
        v = self._unpack_from(1, ">b")
        if v == 0:
            return False
        if v == 1:
            return True
        raise RuntimeError("Unexpected byte for Boolean type")

    def writeBool(self, v):
        self._pack_into(1, ">b", 0 if v else 1)

    {% when Type::String -%}
    # The primitive string type.
    # These write out as size-prefixed utf-8 bytes.

    def readString(self):
        size = self._unpack_from(4, ">I")
        utf8Bytes = self._read_bytes(size)
        return utf8Bytes.decode("utf-8")

    def writeString(self, v):
        utf8Bytes = v.encode("utf-8")
        self._pack_into(4, ">I", len(utf8Bytes))
        self._write_bytes(utf8Bytes)

    {% when Type::Object with (object_name) -%}
    # The Object type {{ object_name }}.
    # Objects cannot currently be serialized, but we can produce a helpful error.

    def read{{ type_name|class_name_py }}(self):
        raise RuntimeError("RustBufferStream.read() not implemented yet for {{ type_name }}")

    def write{{ type_name|class_name_py }}(self):
        raise RuntimeError("RustBufferStream.write() not implemented yet for {{ type_name }}")

    {% when Type::Error with (error_name) -%}
    # The Error type {{ error_name }}.
    # Errors cannot currently be serialized, but we can produce a helpful error.

    def read{{ type_name|class_name_py }}(self):
        raise RuntimeError("RustBufferStream.read() not implemented yet for {{ type_name }}")

    def write{{ type_name|class_name_py }}(self):
        raise RuntimeError("RustBufferStream.write() not implemented yet for {{ type_name }}")

    {% when Type::Enum with (enum_name) -%}
    # The Enum type {{ enum_name }}.

    def read{{ type_name|class_name_py }}(self):
        return {{ enum_name|class_name_py }}(
            self._unpack_from(4, ">I")
        )

    def write{{ type_name|class_name_py }}(self, v):
        self._pack_into(4, ">I", v.value)

    {% when Type::Record with (record_name) -%}
    # The Record type {{ record_name }}.

    {% let rec = ci.get_record_definition(record_name).unwrap() %}

    def read{{ type_name|class_name_py }}(self):
        return {{ rec.name()|class_name_py }}(
            {%- for field in rec.fields() %}
            self.read{{ field.type_().canonical_name()|class_name_py }}(){% if loop.last %}{% else %},{% endif %}
            {%- endfor %}
        )

    def write{{ type_name|class_name_py }}(self, v):
        {%- for field in rec.fields() %}
        self.write{{ field.type_().canonical_name()|class_name_py }}(v.{{ field.name() }})
        {%- endfor %}

    {% when Type::Optional with (inner_type) -%}
    # The Optional<T> type for {{ inner_type.canonical_name() }}.

    def read{{ type_name|class_name_py }}(self):
        flag = self._unpack_from(1, ">B")
        if flag == 0:
            return None
        elif flag == 1:
            return self.read{{ inner_type.canonical_name()|class_name_py }}()
        else:
            raise RuntimeError("Unexpected flag byte for {{ type_name }}")

    def write{{ type_name|class_name_py }}(self, v):
        if v is None:
            self._pack_into(1, ">B", 0)
        else:
            self._pack_into(1, ">B", 1)
            self.write{{ inner_type.canonical_name()|class_name_py }}(v)

    {% when Type::Sequence with (inner_type) -%}
    # The Sequence<T> type for {{ inner_type.canonical_name() }}.

    def read{{ type_name|class_name_py }}(self):
        count = self._unpack_from(4, ">I")
        items = []
        while count > 0:
            items.append(self.read{{ inner_type.canonical_name()|class_name_py }}())
            count -= 1
        return items

    def write{{ type_name|class_name_py }}(self, items):
        # XXX TODO: we should throw if the length is too big for a UInt32.
        self._pack_into(4, ">I", len(items))
        for item in items:
            self.write{{ inner_type.canonical_name()|class_name_py }}(item)

    {% when Type::Map with (inner_type) -%}
    # The Map<T> type for {{ inner_type.canonical_name() }}.

    def read{{ type_name|class_name_py }}(self):
        count = self._unpack_from(4, ">I")
        items = {}
        while count > 0:
            key = self.readString()
            items[key] = self.read{{ inner_type.canonical_name()|class_name_py }}()
            count -= 1
        return items

    def write{{ type_name|class_name_py }}(self, items):
        # XXX TODO: we should throw if the length is too big for a UInt32.
        self._pack_into(4, ">I", len(items))
        for (k, v) in items.items():
            self.writeString(k)
            self.write{{ inner_type.canonical_name()|class_name_py }}(v)

    {%- endmatch %}
    {% endfor %}