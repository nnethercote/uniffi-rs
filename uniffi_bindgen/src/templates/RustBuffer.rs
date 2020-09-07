// Everybody gets basic buffer support, since it's needed for passing complex types over the FFI.

/// This helper allocates a new byte buffer owned by the rust code, and returns it
/// to the foreign-language code as a `RustBuffer` struct. Callers must eventually
/// free the resulting buffer, either by explicitly calling the destructor defined below,
/// or by passing ownership of the buffer back into rust code.
#[no_mangle]
pub extern "C" fn {{ ci.ffi_rustbuffer_alloc().name() }}(size: i32, err: &mut uniffi::deps::ffi_support::ExternError) -> uniffi::RustBuffer {
    uniffi::deps::ffi_support::call_with_output(err, || {
        uniffi::RustBuffer::new_with_size(size.max(0) as usize)
    })
}

/// This helper copies bytes owned by the foreign-language code into a new byte buffer owned
/// by the rust code, and returns it as a `RustBuffer` struct. Callers must eventually
/// free the resulting buffer, either by explicitly calling the destructor defined below,
/// or by passing ownership of the buffer back into rust code.
///
/// # Safety
/// This function will dereference a provided pointer in order to copy bytes from it, so
/// make sure the `ForeignBytes` struct contains a valid pointer and length.
#[no_mangle]
pub unsafe extern "C" fn {{ ci.ffi_rustbuffer_from_bytes().name() }}(bytes: uniffi::ForeignBytes, err: &mut uniffi::deps::ffi_support::ExternError) -> uniffi::RustBuffer {
    uniffi::deps::ffi_support::call_with_output(err, || {
        let bytes = bytes.as_slice();
        uniffi::RustBuffer::from_vec(bytes.to_vec())
    })
}

/// Free a byte buffer that had previously been passed to the foreign language code.
///
/// # Safety
/// The argument *must* be a uniquely-owned `RustBuffer` previously obtained from a call
/// into the rust code that returned a buffer, or you'll risk freeing unowned memory or
/// corrupting the allocator state.
#[no_mangle]
pub unsafe extern "C" fn {{ ci.ffi_rustbuffer_free().name() }}(buf: uniffi::RustBuffer, err: &mut uniffi::deps::ffi_support::ExternError) {
    uniffi::deps::ffi_support::call_with_output(err, || {
        uniffi::RustBuffer::destroy(buf)
    })
}

/// Reserve additional capacity in a byte buffer that had previously been passed to the
/// foreign language code.
///
/// The first argument *must* be a uniquely-owned `RustBuffer` previously
/// obtained from a call into the rust code that returned a buffer. Its underlying data pointer
/// will be reallocated if necessary and returned in a new `RustBuffer` struct.
///
/// The second argument must be the minimum number of *additional* bytes to reserve
/// capacity for in the buffer; it is likely to reserve additional capacity in practice
/// due to amortized growth strategy of rust vectors.
///
/// # Safety
/// The first argument *must* be a uniquely-owned `RustBuffer` previously obtained from a call
/// into the rust code that returned a buffer, or you'll risk freeing unowned memory or
/// corrupting the allocator state.
#[no_mangle]
pub unsafe extern "C" fn {{ ci.ffi_rustbuffer_reserve().name() }}(buf: uniffi::RustBuffer, additional: i32, err: &mut uniffi::deps::ffi_support::ExternError) -> uniffi::RustBuffer {
    uniffi::deps::ffi_support::call_with_output(err, || {
        use std::convert::TryInto;
        let additional: usize = additional.try_into().expect("additional buffer length negative or overflowed");
        let mut v = buf.destroy_into_vec();
        v.reserve(additional);
        uniffi::RustBuffer::from_vec(v)
    })
}