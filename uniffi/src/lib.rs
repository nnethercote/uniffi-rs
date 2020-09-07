/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! # Runtime support code for uniffi
//!
//! This crate provides the small amount of runtime code that is required by the generated uniffi
//! component scaffolding in order to transfer data back and forth across the C-style FFI layer,
//! as well as some utilities for testing the generated bindings.
//!
//! The key concept here is the [`ViaFfi`] trait, which must be implemented for any type that can
//! be passed across the FFI, and which determines:
//!
//!  * How to [represent](ViaFfi::Value) values of that type in the low-level C-style type
//!    system of the FFI layer.
//!  * How to ["lower"](ViaFfi::lower) rust values of that type into an appropriate low-level
//!    FFI value.
//!  * How to ["lift"](ViaFfi::lift) low-level FFI values back into rust values of that type.
//!  * How to [write](ViaFfi::write) rust values of that type into a buffer, for cases
//!    where they are part of a compount data structure that is serialized for transfer.
//!  * How to [read](ViaFfi::read) rust values of that type from buffer, for cases
//!    where they are received as part of a compound data structure that was serialized for transfer.
//!
//! This logic encapsulates the rust-side handling of data transfer. Each foreign-language binding
//! must also implement a matching set of data-handling rules for each data type.
//!
//! In addition to the core` ViaFfi` trait, we provide a handful of struct definitions useful
//! for passing core rust types over the FFI, such as [`RustBuffer`].

use anyhow::{bail, Result};
use bytes::buf::{Buf, BufMut};
use paste::paste;
use std::{
    collections::HashMap,
    convert::{TryFrom, TryInto},
    ffi::CString,
};

// It would be nice if this module was behind a cfg(test) guard, but it
// doesn't work between crates so let's hope LLVM tree-shaking works well.
pub mod testing;

// Re-export the libs that we use in the generated code,
// so the consumer doesn't have to depend on them directly.
pub mod deps {
    pub use anyhow;
    pub use bytes;
    pub use ffi_support;
    pub use lazy_static;
    pub use log;
}

/// Trait defining how to transfer values via the FFI layer.
///
/// The `ViaFfi` trait defines how to pass values of a particular type back-and-forth over
/// the uniffi generated FFI layer, both as standalone argument or return values, and as
/// part of serialized compound data structures.
///
/// (This trait is Like the `InfoFfi` trait from `ffi_support`, but local to this crate
/// so that we can add some alternative implementations for different builtin types,
/// and so that we can add support for receiving as well as returning).
///
/// ## Safety
///
/// This is an unsafe trait (implementing it requires `unsafe impl`) because we can't guarantee
/// that it's safe to pass your type out to foreign-language code and back again. Buggy
/// implementations of this trait might violate some assumptions made by the generated code,
/// or might not match with the corresponding code in the generated foreign-language bindings.
///
/// In general, you should not need to implement this trait by hand, and should instead rely on
/// implementations generated from your component IDL via the `uniffi-bindgen scaffolding` command.

pub unsafe trait ViaFfi: Sized {
    /// The low-level type used for passing values of this type over the FFI.
    ///
    /// This must be a C-compatible type (e.g. a numeric primitive, a `#[repr(C)]` struct) into
    /// which values of the target rust type can be converted.
    ///
    /// For complex data types, we currently recommend using `RustBuffer` and serializing
    /// the data for transfer. In theory it could be possible to build a matching
    /// `#[repr(C)]` struct for a complex data type and pass that instead, but explicit
    /// serialization is simpler and safer as a starting point.
    type FfiType;

    /// Lower a rust value of the target type, into an FFI value of type Self::FfiType.
    ///
    /// This trait method is used for sending data from rust to the foreign language code,
    /// by (hopefully cheaply!) converting it into someting that can be passed over the FFI
    /// and reconstructed on the other side.
    ///
    /// Note that this method takes an owned `self`; this allows it to transfer ownership
    /// in turn to the foreign language code, e.g. by boxing the value and passing a pointer.
    fn lower(self) -> Self::FfiType;

    /// Lift a rust value of the target type, from an FFI value of type Self::FfiType.
    ///
    /// This trait method is used for receiving data from the foreign language code in rust,
    /// by (hopefully cheaply!) converting it from a low-level FFI value of type Self::FfiType
    /// into a high-level rust value of the target type.
    ///
    /// Since we cannot statically guarantee that the foreign-language code will send valid
    /// values of type Self::FfiType, this method is fallible.
    fn try_lift(v: Self::FfiType) -> Result<Self>;

    /// Write a rust value into a buffer, to send over the FFI in serialized form.
    ///
    /// This trait method can be used for sending data from rust to the foreign language code,
    /// in cases where we're not able to use a special-purpose FFI type and must fall back to
    /// sending serialized bytes.
    fn write<B: BufMut>(&self, buf: &mut B);

    /// Read a rust value from a buffer, received over the FFI in serialized form.
    ///
    /// This trait method can be used for receiving data from the foreign language code in rust,
    /// in cases where we're not able to use a special-purpose FFI type and must fall back to
    /// receiving serialized bytes.
    ///
    /// Since we cannot statically guarantee that the foreign-language code will send valid
    /// serialized bytes for the target type, this method is fallible.
    fn try_read<B: Buf>(buf: &mut B) -> Result<Self>;
}

/// A helper function to lower a type by serializing it into a buffer.
///
/// For complex types were it's too fiddly or too unsafe to convert them into a special-purpose
/// C-compatible value, you can use this helper function to implement `lower()` in terms of `write()`
/// and pass the value as a serialized buffer of bytes.
pub fn lower_into_buffer<T: ViaFfi>(value: T) -> RustBuffer {
    let mut buf = Vec::new();
    ViaFfi::write(&value, &mut buf);
    RustBuffer::from_vec(buf)
}

/// A helper function to lift a type by deserializing it from a buffer.
///
/// For complex types were it's too fiddly or too unsafe to convert them into a special-purpose
/// C-compatible value, you can use this helper function to implement `lift()` in terms of `read()`
/// and receive the value as a serialzied byte buffer.
pub fn try_lift_from_buffer<T: ViaFfi>(buf: RustBuffer) -> Result<T> {
    let vec = buf.destroy_into_vec();
    let mut buf = vec.as_slice();
    let value = <T as ViaFfi>::try_read(&mut buf)?;
    if buf.remaining() != 0 {
        bail!("junk data left in buffer after lifting")
    }
    Ok(value)
}

/// A helper function to ensure we don't read past the end of a buffer.
///
/// Rust won't actually let us read past the end of a buffer, but the `Buf` trait does not support
/// returning an explicit error in this case, and will instead panic. This is a look-before-you-leap
/// helper function to instead return an explicit error, to help with debugging.
pub fn check_remaining<B: Buf>(buf: &B, num_bytes: usize) -> Result<()> {
    if buf.remaining() < num_bytes {
        bail!("not enough bytes remaining in buffer");
    }
    Ok(())
}

/// Blanket implementation of ViaFfi for numeric primitives.
///
/// Numeric primitives have a straightforward mapping into C-compatible numeric types,
/// sice they are themselves a C-compatible numeric type!
macro_rules! impl_via_ffi_for_num_primitive {
    ($($T:ty,)+) => { impl_via_ffi_for_num_primitive!($($T),+); };
    ($($T:ty),*) => {
            $(
                paste! {
                    unsafe impl ViaFfi for $T {
                        type FfiType = Self;

                        fn lower(self) -> Self::FfiType {
                            self
                        }

                        fn try_lift(v: Self::FfiType) -> Result<Self> {
                            Ok(v)
                        }

                        fn write<B: BufMut>(&self, buf: &mut B) {
                            buf.[<put_ $T>](*self);
                        }

                        fn try_read<B: Buf>(buf: &mut B) -> Result<Self> {
                            check_remaining(buf, std::mem::size_of::<$T>())?;
                            Ok(buf.[<get_ $T>]())
                        }
                    }
                }
            )*
    };
}

impl_via_ffi_for_num_primitive! {
    i8, u8, i16, u16, i32, u32, i64, u64, f32, f64
}

/// Support for passing boolean values via the FFI.
///
/// Booleans are passed as a `u8` in order to avoid problems with handling
/// C-compatible boolean values on JVM-based languages.
unsafe impl ViaFfi for bool {
    type FfiType = u8;

    fn lower(self) -> Self::FfiType {
        if self {
            1
        } else {
            0
        }
    }

    fn try_lift(v: Self::FfiType) -> Result<Self> {
        Ok(match v {
            0 => false,
            1 => true,
            _ => bail!("unexpected byte for Boolean"),
        })
    }

    fn write<B: BufMut>(&self, buf: &mut B) {
        buf.put_u8(ViaFfi::lower(*self));
    }

    fn try_read<B: Buf>(buf: &mut B) -> Result<Self> {
        check_remaining(buf, 1)?;
        ViaFfi::try_lift(buf.get_u8())
    }
}

/// Support for passing a buffer of bytes via the FFI.
///
/// We can pass a `Vec<u8>` to foreign language code by decomposing it into
/// its raw parts (buffer pointer, length, and capacity) and passing those
/// around as a struct. Naturally, this can be tremendously unsafe! So here
/// are the details:
///
///   * `RustBuffer` structs must only ever be constructed from a `Vec<u8>`,
///     either explicitly via `RustBuffer::from_vec` or indirectly by calling
///     one of the `RustBuffer::new*` constructors.
///
///   * `RustBuffer` structs do not implement `Drop`, since they are intended
///     to be passed to foreign-language code outside of the control of rust's
///     ownership system. To avoid memory leaks they *must* passed back into
///     rust and either explicitly destroyed using `RustBuffer::destroy`, or
///     converted back to a `Vec<u8>` using `RustBuffer::destroy_into_vec`
///     (which will then be dropped via rust's usual ownership-tracking system).
///
/// Implementation note: all the fields of this struct are private, so you can't
/// manually construct instances that don't come from a `Vec<u8>`. If you've got
/// a `RustBuffer` then it either came from a public constructor (all of which
/// are safe) or it came from foreign-language code (which should be encforcing
/// safety invariants).
///
/// This struct is based on `ByteBuffer` from the `ffi-support` crate, but modified
/// to retain unallocated capacity rather than truncating to the occupied length.
#[repr(C)]
pub struct RustBuffer {
    /// The allocated capacity of the underlying `Vec<u8>`.
    /// In rust this is a `usize`, but we use an `i32` for compatibility with JNA.
    capacity: i32,
    /// The occupied length of the underlying `Vec<u8>`.
    /// In rust this is a `usize`, but we use an `i32` for compatibility with JNA.
    len: i32,
    /// The pointer to the allocated buffer of the `Vec<u8>`.
    data: *mut u8,
}

impl RustBuffer {
    /// Creates an empty `RustBuffer`.
    ///
    /// The resulting vector will not be automatically dropped; you must
    /// arrange to call `destroy` or `destroy_into_vec` when finished with it.
    pub fn new() -> Self {
        Self::from_vec(Vec::new())
    }

    pub fn len(&self) -> usize {
        self.len
            .try_into()
            .expect("buffer length negative or overflowed")
    }

    /// Creates a `RustBuffer` zero-filed to the requested size.
    ///
    /// The resulting vector will not be automatically dropped; you must
    /// arrange to call `destroy` or `destroy_into_vec` when finished with it.
    pub fn new_with_size(size: usize) -> Self {
        // Note: `Vec` requires this internally on 64 bit platforms (and has a
        // stricter requirement on 32 bit ones), so this is just to be explicit.
        assert!(size < i64::MAX as usize);
        let mut buf = vec![];
        buf.resize(size, 0);
        Self::from_vec(buf)
    }

    /// Consumes a `Vec<u8>` and returns its raw parts as a `RustBuffer`.
    ///
    /// The resulting vector will not be automatically dropped; you must
    /// arrange to call `destroy` or `destroy_into_vec` when finished with it.
    pub fn from_vec(v: Vec<u8>) -> Self {
        let mut v = std::mem::ManuallyDrop::new(v);
        Self {
            capacity: i32::try_from(v.capacity()).expect("buffer capacity cannot fit into a i32."),
            len: i32::try_from(v.len()).expect("buffer length cannot fit into a i32."),
            data: v.as_mut_ptr(),
        }
    }

    /// Converts this `RustBuffer` back into an owned `Vec<u8>`.
    ///
    /// This restores ownership of the underlying buffer to rust, meaning it will
    /// be dropped when the `Vec<u8>` is dropped. The `RustBuffer` *must* have been
    /// previously obtained from a valid `Vec<u8>` owned by this rust code.
    pub fn destroy_into_vec(self) -> Vec<u8> {
        if self.data.is_null() {
            assert!(self.capacity == 0, "null RustBuffer had non-zero capacity");
            assert!(self.len == 0, "null RustBuffer had non-zero length");
            vec![]
        } else {
            let capacity: usize = self
                .capacity
                .try_into()
                .expect("buffer capacity negative or overflowed");
            let len: usize = self
                .len
                .try_into()
                .expect("buffer length negative or overflowed");
            unsafe { Vec::from_raw_parts(self.data, len, capacity) }
        }
    }

    /// Reclaim memory stored in this `RustBuffer`.
    pub fn destroy(self) {
        drop(self.destroy_into_vec());
    }
}

unsafe impl deps::ffi_support::IntoFfi for RustBuffer {
    type Value = Self;
    fn ffi_default() -> Self {
        Self {
            capacity: 0,
            len: 0,
            data: std::ptr::null_mut(),
        }
    }
    fn into_ffi_value(self) -> Self::Value {
        self
    }
}

/// Support for reading a slice of bytes provided over the FFI.
///
/// Foreign language code can pass a slice of bytes by providing a data pointer
/// and length, and this struct provides a convenient wrapper for working with
/// that pair. Naturally, this can be tremendously unsafe! So here are the details:
///
///   * The foreign language code must ensure the provided buffer stays alive
///     and unchanged for the duration of the call to which the `ForeignBytes`
///     struct was provided.
///
/// To work with the bytes form rust code, use `as_slice()` to view the data
/// as a `&[u8]`.
///
/// Implementation note: all the fields of this struct are private and it has no
/// constructors, so consuming crates cant create invalid instance of it. If you've
/// got a `ForeignBytes`, then you received it over the FFI and are assuming that
/// the foreign language code is upholding the above invariants.
///
/// This struct is based on `ByteBuffer` from the `ffi-support` crate, but modified
/// to give a read-only view of externally-provided bytes.
#[repr(C)]
pub struct ForeignBytes {
    /// The length of the pointed-to data.
    /// We use an `i32` for compatibility with JNA.
    len: i32,
    /// The pointer to the foreign-owned bytes.
    data: *const u8,
}

impl ForeignBytes {
    /// View the foreign bytes as a `&[u8]`.
    pub fn as_slice<'a>(&'a self) -> &'a [u8] {
        if self.data.is_null() {
            assert!(self.len == 0, "null ForeignBytes had non-zero length");
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.data, self.len()) }
        }
    }

    /// Get the length of this slice of bytes.
    pub fn len(&self) -> usize {
        self.len
            .try_into()
            .expect("bytes length negative or overflowed")
    }
}

/// Support for passing optional values via the FFI.
///
/// Optional values are currently always passed by serializing to a buffer.
/// We write either a zero byte for `None`, or a one byte followed by the containing
/// item for `Some`.
///
/// In future we could do the same optimization as rust uses internally, where the
/// `None` option is represented as a null pointer and the `Some` as a valid pointer,
/// but that seems more fiddly and less safe in the short term, so it can wait.
unsafe impl<T: ViaFfi> ViaFfi for Option<T> {
    type FfiType = RustBuffer;

    fn lower(self) -> Self::FfiType {
        lower_into_buffer(self)
    }

    fn try_lift(v: Self::FfiType) -> Result<Self> {
        try_lift_from_buffer(v)
    }

    fn write<B: BufMut>(&self, buf: &mut B) {
        match self {
            None => buf.put_u8(0),
            Some(v) => {
                buf.put_u8(1);
                ViaFfi::write(v, buf);
            }
        }
    }

    fn try_read<B: Buf>(buf: &mut B) -> Result<Self> {
        check_remaining(buf, 1)?;
        Ok(match buf.get_u8() {
            0 => None,
            1 => Some(<T as ViaFfi>::try_read(buf)?),
            _ => bail!("unexpected tag byte for Option"),
        })
    }
}

/// Support for passing vectors of values via the FFI.
///
/// Vectors are currently always passed by serializing to a buffer.
/// We write a `u32` item count followed by each item in turn.
///
/// Ideally we would pass `Vec<u8>` directly as a `RustBuffer` rather
/// than serializing, and perhaps even pass other vector types using a
/// similar struct. But that's for future work.
unsafe impl<T: ViaFfi> ViaFfi for Vec<T> {
    type FfiType = RustBuffer;

    fn lower(self) -> Self::FfiType {
        lower_into_buffer(self)
    }

    fn try_lift(v: Self::FfiType) -> Result<Self> {
        try_lift_from_buffer(v)
    }

    fn write<B: BufMut>(&self, buf: &mut B) {
        // TODO: would be nice not to panic here :-/
        let len = u32::try_from(self.len()).unwrap();
        buf.put_u32(len); // We limit arrays to u32::MAX items
        for item in self.iter() {
            ViaFfi::write(item, buf);
        }
    }

    fn try_read<B: Buf>(buf: &mut B) -> Result<Self> {
        check_remaining(buf, 4)?;
        let len = buf.get_u32();
        let mut vec = Vec::with_capacity(len as usize);
        for _ in 0..len {
            vec.push(<T as ViaFfi>::try_read(buf)?)
        }
        Ok(vec)
    }
}

/// Support for associative arrays via the FFI.
/// Note that because of webidl limitations,
/// the key must always be of the String type.
///
/// HashMaps are currently always passed by serializing to a buffer.
/// We write a `u32` entries count followed by each entry (string
/// key followed by the value) in turn.
unsafe impl<V: ViaFfi> ViaFfi for HashMap<String, V> {
    type FfiType = RustBuffer;

    fn lower(self) -> Self::FfiType {
        lower_into_buffer(self)
    }

    fn try_lift(v: Self::FfiType) -> Result<Self> {
        try_lift_from_buffer(v)
    }

    fn write<B: BufMut>(&self, buf: &mut B) {
        // TODO: would be nice not to panic here :-/
        let len = u32::try_from(self.len()).unwrap();
        buf.put_u32(len); // We limit HashMaps to u32::MAX entries
        for (key, value) in self.iter() {
            ViaFfi::write(key, buf);
            ViaFfi::write(value, buf);
        }
    }

    fn try_read<B: Buf>(buf: &mut B) -> Result<Self> {
        check_remaining(buf, 4)?;
        let len = buf.get_u32();
        let mut map = HashMap::with_capacity(len as usize);
        for _ in 0..len {
            let key = String::try_read(buf)?;
            let value = <V as ViaFfi>::try_read(buf)?;
            map.insert(key, value);
        }
        Ok(map)
    }
}

/// Support for passing Strings via the FFI.
///
/// Unlike many other implementations of `ViaFfi`, this passes a pointer rather
/// than copying the data from one side to the other. This is a safety hazard,
/// but turns out to be pretty nice for useability. This pointer *must be one owned
/// by the rust allocator and it *must* point to valid utf-8 data (in other words,
/// it *must* be an actual rust `String`).
///
/// When serialized in a bytebuffer, strings are represented as a u32 byte length
/// followed by utf8-encoded bytes.
///
/// (In practice, we currently do end up copying the data, the copying just happens
/// on the foreign language side rather than here in the rust code.)
unsafe impl ViaFfi for String {
    type FfiType = *mut std::os::raw::c_char;

    // This returns a raw pointer to the underlying bytes, so it's very important
    // that it consume ownership of the String, which is relinquished to the foreign
    // language code (and can be restored by it passing the pointer back).
    fn lower(self) -> Self::FfiType {
        ffi_support::rust_string_to_c(self)
    }

    // The argument here *must* be a uniquely-owned pointer previously obtained
    // from `info_ffi_value` above. It will try to panic if you give it an invalid
    // pointer, but there's no guarantee that it will.
    fn try_lift(v: Self::FfiType) -> Result<Self> {
        if v.is_null() {
            bail!("null pointer passed as String")
        }
        let cstr = unsafe { CString::from_raw(v) };
        // This turns the buffer back into a `String` without copying the data
        // and without re-checking it for validity of the utf8. If the pointer
        // came from a valid String then there's no point in re-checking the utf8,
        // and if it didn't then bad things are going to happen regardless of
        // whether we check for valid utf8 data or not.
        Ok(unsafe { String::from_utf8_unchecked(cstr.into_bytes()) })
    }

    fn write<B: BufMut>(&self, buf: &mut B) {
        // N.B. `len()` gives us the length in bytes, not in chars or graphemes.
        // TODO: it would be nice not to panic here.
        let len = u32::try_from(self.len()).unwrap();
        buf.put_u32(len); // We limit strings to u32::MAX bytes
        buf.put(self.as_bytes());
    }

    fn try_read<B: Buf>(buf: &mut B) -> Result<Self> {
        check_remaining(buf, 4)?;
        let len = buf.get_u32() as usize;
        check_remaining(buf, len)?;
        let bytes = &buf.bytes()[..len];
        let res = String::from_utf8(bytes.to_vec())?;
        buf.advance(len);
        Ok(res)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_rustbuffer_from_vec() {
        let rbuf = RustBuffer::from_vec(vec![1u8, 2, 3]);
        assert_eq!(rbuf.len(), 3);
        assert_eq!(rbuf.destroy_into_vec(), vec![1u8, 2, 3]);
    }

    #[test]
    fn test_rustbuffer_empty() {
        let rbuf = RustBuffer::new();
        assert_eq!(rbuf.len(), 0);
        assert!(!rbuf.data.is_null());
        assert_eq!(rbuf.destroy_into_vec(), Vec::<u8>::new());
    }

    #[test]
    fn test_rustbuffer_new_with_size() {
        let rbuf = RustBuffer::new_with_size(5);
        assert_eq!(rbuf.destroy_into_vec().as_slice(), &[0u8, 0, 0, 0, 0]);

        let rbuf = RustBuffer::new_with_size(0);
        assert!(!rbuf.data.is_null());
        assert_eq!(rbuf.destroy_into_vec().as_slice(), &[0u8; 0]);
    }

    #[test]
    fn test_rustbuffer_null_means_empty() {
        let rbuf = RustBuffer {
            capacity: 0,
            len: 0,
            data: std::ptr::null_mut(),
        };
        assert_eq!(rbuf.destroy_into_vec().as_slice(), &[0u8; 0]);
    }

    #[test]
    #[should_panic]
    fn test_rustbuffer_null_must_have_no_capacity() {
        let rbuf = RustBuffer {
            capacity: 1,
            len: 0,
            data: std::ptr::null_mut(),
        };
        rbuf.destroy_into_vec();
    }
    #[test]
    #[should_panic]
    fn test_rustbuffer_null_must_have_zero_length() {
        let rbuf = RustBuffer {
            capacity: 0,
            len: 12,
            data: std::ptr::null_mut(),
        };
        rbuf.destroy_into_vec();
    }

    #[test]
    #[should_panic]
    fn test_rustbuffer_provided_capacity_must_be_non_negative() {
        let mut v = vec![0u8, 1, 2];
        let rbuf = RustBuffer {
            capacity: -7,
            len: 3,
            data: v.as_mut_ptr(),
        };
        rbuf.destroy_into_vec();
    }

    #[test]
    #[should_panic]
    fn test_rustbuffer_provided_len_must_be_non_negative() {
        let mut v = vec![0u8, 1, 2];
        v.shrink_to_fit();
        let rbuf = RustBuffer {
            capacity: 3,
            len: -1,
            data: v.as_mut_ptr(),
        };
        rbuf.destroy_into_vec();
    }

    #[test]
    #[should_panic]
    fn test_rustbuffer_vec_capacity_must_fit_in_i32() {
        RustBuffer::from_vec(Vec::with_capacity((i32::MAX as usize) + 1));
    }

    #[test]
    #[should_panic]
    fn test_rustbuffer_vec_len_must_fit_in_i32() {
        let mut v = Vec::new();
        // We don't want to actually materialize a huge vec, so unsafety it is!
        // This won't cause problems because the contained items are Plain Old Data.
        // (And also we expect to panic without accessing them).
        unsafe { v.set_len((i32::MAX as usize) + 1) }
        RustBuffer::from_vec(v);
    }

    #[test]
    fn test_foreignbytes_access() {
        let rbuf = RustBuffer::from_vec(vec![1u8, 2, 3]);
        let fbuf = ForeignBytes {
            len: rbuf.len,
            data: rbuf.data,
        };
        assert_eq!(fbuf.len(), 3);
        assert_eq!(fbuf.as_slice(), &[1u8, 2, 3]);
        rbuf.destroy()
    }

    #[test]
    fn test_foreignbytes_empty() {
        let rbuf = RustBuffer::new();
        let fbuf = ForeignBytes {
            len: rbuf.len,
            data: rbuf.data,
        };
        assert_eq!(fbuf.len(), 0);
        assert_eq!(fbuf.as_slice(), &[0u8; 0]);
        rbuf.destroy()
    }

    #[test]
    fn test_foreignbytes_null_means_empty() {
        let fbuf = ForeignBytes {
            len: 0,
            data: std::ptr::null_mut(),
        };
        assert_eq!(fbuf.as_slice(), &[0u8; 0]);
    }

    #[test]
    #[should_panic]
    fn test_foreignbytes_null_must_have_zero_length() {
        let fbuf = ForeignBytes {
            len: 12,
            data: std::ptr::null_mut(),
        };
        fbuf.as_slice();
    }

    #[test]
    #[should_panic]
    fn test_foreignbytes_provided_len_must_be_non_negative() {
        let mut v = vec![0u8, 1, 2];
        let fbuf = ForeignBytes {
            len: -1,
            data: v.as_mut_ptr(),
        };
        fbuf.as_slice();
    }
}
