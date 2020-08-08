# This is a helper for safely working with byte buffers returned from the Rust code.
# It's basically a wrapper around a length and a data pointer, corresponding to the
# `ffi_support::ByteBuffer` struct on the rust side.

class RustBuffer(ctypes.Structure):
    _fields_ = [
        ("len", ctypes.c_long),
        ("data", ctypes.POINTER(ctypes.c_char)),
    ]

    @staticmethod
    def alloc(size):
        return _UniFFILib.{{ ci.ffi_bytebuffer_alloc().name() }}(size)

    def free(self):
        return _UniFFILib.{{ ci.ffi_bytebuffer_free().name() }}(self)

    def __str__(self):
        return "RustBuffer(len={}, data={})".format(self.len, self.data[0:self.len])

    # For every type that lowers into a RustBuffer, we provide helper methods for
    # conveniently doing the lifting and lowering. Putting them on this internal
    # helper object (rather than, say, as methods on the public classes) makes it
    # easier for us to hide these implementation details from consumers, in the face
    # of python's free-for-all type system.

    {% for typ in ci.iter_types() %}
    {% let type_name = typ.canonical_name() %}
    {%- match typ -%}

    {% when Type::Record with (record_name) -%}
    # The Record type {{ record_name }}.

    {% let rec = ci.get_record_definition(record_name).unwrap() %}

    @classmethod
    def allocFrom{{ type_name|class_name_py }}(cls, v):
        size = cls.calculateWriteSizeOf{{ type_name|class_name_py }}(v)
        self = cls.alloc(size)
        try:
            RustBufferStream(self).write{{ type_name|class_name_py }}(v)
        except Exception:
            self.free()
            raise
        return self

    @staticmethod
    def calculateWriteSizeOf{{ type_name|class_name_py }}(v):
        return 0 + \
            {%- for field in rec.fields() %}
            + {{ "(v.{})"|format(field.name())|calculate_write_size(field.type_()) }} \
            {%- endfor %}

    def consumeInto{{ type_name|class_name_py }}(self):
        try:
            buf = RustBufferStream(self)
            v = buf.read{{ type_name|class_name_py }}()
            if buf.remaining() != 0:
                raise RuntimeError("junk data left in buffer after lifting a {{ type_name }}")
            return v
        finally:
            self.free()

    {% when Type::Optional with (inner_type) -%}
    # The Optional<T> type for {{ inner_type.canonical_name() }}.

    @classmethod
    def allocFrom{{ type_name|class_name_py }}(cls, v):
        size = cls.calculateWriteSizeOf{{ type_name|class_name_py }}(v)
        self = cls.alloc(size)
        try:
            RustBufferStream(self).write{{ type_name|class_name_py }}(v)
        except Exception:
            self.free()
            raise
        return self

    @staticmethod
    def calculateWriteSizeOf{{ type_name|class_name_py }}(v):
        return 1 + (0 if v is None else {{ "v"|calculate_write_size(inner_type) }})

    def consumeInto{{ type_name|class_name_py }}(self):
        try:
            buf = RustBufferStream(self)
            v = buf.read{{ type_name|class_name_py }}()
            if buf.remaining() != 0:
                raise RuntimeError("junk data left in buffer after lifting a {{ type_name }}")
            return v
        finally:
            self.free()

    {% when Type::Sequence with (inner_type) -%}
    # The Sequence<T> type for {{ inner_type.canonical_name() }}.

    @classmethod
    def allocFrom{{ type_name|class_name_py }}(cls, v):
        size = cls.calculateWriteSizeOf{{ type_name|class_name_py }}(v)
        self = cls.alloc(size)
        try:
            RustBufferStream(self).write{{ type_name|class_name_py }}(v)
        except Exception:
            self.free()
            raise
        return self

    @staticmethod
    def calculateWriteSizeOf{{ type_name|class_name_py }}(v):
        return 4 + sum({{ "item"|calculate_write_size(inner_type) }} for item in v)

    def consumeInto{{ type_name|class_name_py }}(self):
        try:
            buf = RustBufferStream(self)
            v = buf.read{{ type_name|class_name_py }}()
            if buf.remaining() != 0:
                raise RuntimeError("junk data left in buffer after lifting a {{ type_name }}")
            return v
        finally:
            self.free()

    {% when Type::Map with (inner_type) -%}
    # The Map<T> type for {{ inner_type.canonical_name() }}.

    @classmethod
    def allocFrom{{ type_name|class_name_py }}(cls, v):
        size = cls.calculateWriteSizeOf{{ type_name|class_name_py }}(v)
        self = cls.alloc(size)
        try:
            RustBufferStream(self).write{{ type_name|class_name_py }}(v)
        except Exception:
            self.free()
            raise
        return self

    @staticmethod
    def calculateWriteSizeOf{{ type_name|class_name_py }}(value):
        return 4 + sum({{ "k"|calculate_write_size(Type::String) }} + {{ "v"|calculate_write_size(inner_type) }} for (k, v) in value.items())

    def consumeInto{{ type_name|class_name_py }}(self):
        try:
            buf = RustBufferStream(self)
            v = buf.read{{ type_name|class_name_py }}()
            if buf.remaining() != 0:
                raise RuntimeError("junk data left in buffer after lifting a {{ type_name }}")
            return v
        finally:
            self.free()
    {% else -%}
    {# No code emitted for types that don't lower into a RustBuffer #}
    {%- endmatch %}
    {% endfor %}