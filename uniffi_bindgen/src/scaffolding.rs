/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use anyhow::Result;
use askama::Template;

use super::interface::*;

#[derive(Template)]
#[template(syntax = "rs", escape = "none", path = "scaffolding_template.rs")]
pub struct RustScaffolding<'a> {
    ci: &'a ComponentInterface,
}
impl<'a> RustScaffolding<'a> {
    pub fn new(ci: &'a ComponentInterface) -> Self {
        Self { ci }
    }
}

mod filters {
    use super::*;
    use std::fmt;

    pub fn type_rs(type_: &Type) -> Result<String, askama::Error> {
        Ok(match type_ {
            Type::Int8 => "i8".into(),
            Type::UInt8 => "u8".into(),
            Type::Int16 => "i16".into(),
            Type::UInt16 => "u16".into(),
            Type::Int32 => "i32".into(),
            Type::UInt32 => "u32".into(),
            Type::Int64 => "i64".into(),
            Type::UInt64 => "u64".into(),
            Type::Float32 => "f32".into(),
            Type::Float64 => "f64".into(),
            Type::Boolean => "bool".into(),
            Type::String => "String".into(),
            Type::Enum(name) | Type::Record(name) | Type::Object(name) | Type::Error(name) => {
                name.clone()
            }
            Type::Optional(t) => format!("Option<{}>", type_rs(t)?),
            Type::Sequence(t) => format!("Vec<{}>", type_rs(t)?),
            Type::Map(t) => format!("std::collections::HashMap<String, {}>", type_rs(t)?),
        })
    }

    pub fn type_ffi(type_: &FFIType) -> Result<String, askama::Error> {
        Ok(match type_ {
            FFIType::Int8 => "i8".into(),
            FFIType::UInt8 => "u8".into(),
            FFIType::Int16 => "i16".into(),
            FFIType::UInt16 => "u16".into(),
            FFIType::Int32 => "i32".into(),
            FFIType::UInt32 => "u32".into(),
            FFIType::Int64 => "i64".into(),
            FFIType::UInt64 => "u64".into(),
            FFIType::Float32 => "f32".into(),
            FFIType::Float64 => "f64".into(),
            FFIType::RustString => "*mut std::os::raw::c_char".into(),
            FFIType::RustBuffer => "uniffi::RustBuffer".into(),
            FFIType::RustError => "uniffi::deps::ffi_support::ExternError".into(),
            FFIType::ForeignStringRef => "*const std::os::raw::c_char".into(),
            FFIType::ForeignBytes => "uniffi::ForeignBytes".into(),
        })
    }

    pub fn lower_rs(nm: &dyn fmt::Display, type_: &Type) -> Result<String, askama::Error> {
        // By explicitly naming the type here, we help the rust compiler to type-check the user-provided
        // implementations of the functions that we're wrapping (and also to type-check our generated code).
        Ok(format!(
            "<{} as uniffi::ViaFfi>::lower({})",
            type_rs(type_)?,
            nm
        ))
    }

    pub fn lift_rs(nm: &dyn fmt::Display, type_: &Type) -> Result<String, askama::Error> {
        // By explicitly naming the type here, we help the rust compiler to type-check the user-provided
        // implementations of the functions that we're wrapping (and also to type-check our generated code).
        // This will panic if the bindings provide an invalid value over the FFI.
        Ok(format!(
            "<{} as uniffi::ViaFfi>::try_lift({}).unwrap()",
            type_rs(type_)?,
            nm
        ))
    }
}
